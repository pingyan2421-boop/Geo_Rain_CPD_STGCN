import argparse, csv, os, shutil, sys, time, warnings, re
from copy import copy
import numpy as np, pandas as pd, tensorflow as tf
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Passing.*as a synonym of type is deprecated")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel("ERROR")
tf.compat.v1.disable_eager_execution()
current_file_path = os.path.abspath(__file__)
models_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(models_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from data_loader.date_loader import build_adjacency_matrix, build_hydro_flow_weight, detect_change_points_professional, load_rain_susceptibility, load_and_clean_data
from data_loader.cpd_methods import CPD_METHODS
from data_loader.rainfall_loader import align_rainfall_to_insar, forecast_context_for_starts, normalized_rain_matrix
from models.base_model import build_model
from scripts.common import prediction_artifacts as artifacts

def make_windows(norm_seq, stage_seq, indices, n_his, n_pred):
    n_frame = n_his + n_pred
    x = np.zeros((len(indices), n_frame, norm_seq.shape[1], 1), dtype=(np.float32))
    s = np.zeros((len(indices), n_his), dtype=(np.int32))
    for row, start in enumerate(indices):
        x[row, :, :, 0] = norm_seq[start:start + n_frame]
        s[row] = stage_seq[start:start + n_his]

    return (
     x, s)


def make_rain_windows(rain_seq, indices, n_his):
    if rain_seq is None:
        return
    else:
        x = np.zeros((len(indices), n_his, rain_seq.shape[1]), dtype=(np.float32))
        for row, start in enumerate(indices):
            x[row] = rain_seq[start:start + n_his]

        return x


def make_aligned_windows(aligned_df, indices, n_his):
    if aligned_df is None:
        return
    else:
        return [aligned_df.iloc[start:start + n_his] for start in indices]


def select_forecast_context(forecast_context, indices):
    if forecast_context is None:
        return
    else:
        return forecast_context[np.asarray(indices, dtype=(np.int32))]


def build_event_sample_weights(event_context, feature_cols, strength=0.0, mode='active'):
    if event_context is None or strength <= 0.0:
        return
    else:
        feature_cols = list(feature_cols or [])
        weights = np.ones((event_context.shape[0], 1, 1, 1), dtype=(np.float32))
        active = np.zeros((event_context.shape[0]), dtype=(np.float32))
        for name, scale in (('event_active_3d', 1.0), ('event_active_7d', 0.7), ('event_active_15d', 0.4),
                            ('event_recent_30d', 0.2)):
            if name in feature_cols:
                active = np.maximum(active, scale * np.clip(event_context[:, feature_cols.index(name)], 0.0, 1.0))

        if "event_antecedent_wetness_index" in feature_cols:
            wet = event_context[:, feature_cols.index("event_antecedent_wetness_index")]
            wet = np.nan_to_num(wet, nan=0.0, posinf=0.0, neginf=0.0)
            if np.max(wet) > np.min(wet):
                wet = (wet - np.min(wet)) / max(np.max(wet) - np.min(wet), 1e-06)
                active = np.maximum(active, 0.25 * wet.astype(np.float32))
        if mode == "probability":
            if "event_historical_trigger_probability" in feature_cols:
                probability = event_context[:, feature_cols.index("event_historical_trigger_probability")]
                probability = np.nan_to_num(probability, nan=0.5, posinf=0.5, neginf=0.0)
                probability = np.clip(probability.astype(np.float32), 0.0, 1.0)
                active = active * probability
        weights[:, 0, 0, 0] += np.float32(strength) * active.astype(np.float32)
        return weights


def _event_level_code(value):
    text = str(value)
    if text.startswith("W3") or text.startswith("P95"):
        return 3.0
    if text.startswith("W2") or text.startswith("P90"):
        return 2.0
    else:
        if text.startswith("W1") or text.startswith("P75"):
            return 1.0
        return 0.0


def add_historical_event_priors(events):
    events = events.sort_values("date").reset_index(drop=True).copy()
    if "is_response_event" in events.columns:
        responses = events["is_response_event"].astype(bool)
    else:
        responses = pd.Series(False, index=(events.index))
    probabilities = []
    counts = []
    for idx, event in events.iterrows():
        past = events.iloc[:idx]
        past_responses = responses.iloc[:idx]
        same_class = pd.Series(False, index=(past.index))
        if "wetness_level" in events.columns:
            if "rain_level" in events.columns:
                same_class = (past["wetness_level"].astype(str) == str(event.get("wetness_level", ""))) & (past["rain_level"].astype(str) == str(event.get("rain_level", "")))
            subset = past_responses[same_class] if same_class.any() else past_responses
            n = int(len(subset))
            probability = (float(subset.sum()) + 1.0) / (float(n) + 2.0) if n > 0 else 0.5
            probabilities.append(probability)
            counts.append(n)

    events["historical_trigger_probability"] = np.asarray(probabilities, dtype=(np.float32))
    events["historical_trigger_count"] = np.asarray(counts, dtype=(np.float32))
    return events


def parse_insar_timestamp(value):
    return pd.to_datetime(re.sub("\\.\\d+$", "", str(value)))


def event_context_for_starts(event_catalog_csv, insar_dates, starts, n_his, lookback_days=60):
    """Build non-leaking sample-level event state from past event features only."""
    events = pd.read_csv(event_catalog_csv)
    if "date" not in events.columns:
        raise ValueError("Event catalog CSV must contain a date column.")
    events = events.copy()
    events["date"] = pd.to_datetime((events["date"]), errors="coerce")
    events = events.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    events = add_historical_event_priors(events)
    dates = pd.to_datetime([parse_insar_timestamp(d) for d in insar_dates])
    feature_cols = [
     'event_available', 
     'event_age_days', 
     'event_active_3d', 
     'event_active_7d', 
     'event_active_15d', 
     'event_recent_30d', 
     'event_rain_mm', 
     'event_spell_rain_sum', 
     'event_pre30_rain_sum', 
     'event_pre60_rain_sum', 
     'event_pre60_p90_days', 
     'event_max_roll30_before', 
     'event_antecedent_wetness_index', 
     'event_wetness_level_code', 
     'event_rain_level_code', 
     'event_selection_score', 
     'event_historical_trigger_probability', 
     'event_historical_trigger_count']
    rows = []
    for start in starts:
        origin_date = pd.Timestamp(dates[int(start) + int(n_his) - 1])
        row = {name: 0.0 for name in feature_cols}
        row["event_age_days"] = float(lookback_days)
        past = events[events["date"] <= origin_date]
        if not past.empty:
            event = past.iloc[-1]
            age = float((origin_date - event["date"]).days)
            if age <= float(lookback_days):
                row["event_available"] = 1.0
                row["event_age_days"] = age
                row["event_active_3d"] = float(age <= 3.0)
                row["event_active_7d"] = float(age <= 7.0)
                row["event_active_15d"] = float(age <= 15.0)
                row["event_recent_30d"] = float(age <= 30.0)
                row["event_rain_mm"] = float(event.get("rain_mm", 0.0))
                row["event_spell_rain_sum"] = float(event.get("spell_rain_sum", 0.0))
                row["event_pre30_rain_sum"] = float(event.get("pre30_rain_sum", 0.0))
                row["event_pre60_rain_sum"] = float(event.get("pre60_rain_sum", 0.0))
                row["event_pre60_p90_days"] = float(event.get("pre60_p90_days", 0.0))
                row["event_max_roll30_before"] = float(event.get("max_roll30_before", 0.0))
                row["event_antecedent_wetness_index"] = float(event.get("antecedent_wetness_index", 0.0))
                row["event_wetness_level_code"] = _event_level_code(event.get("wetness_level", ""))
                row["event_rain_level_code"] = _event_level_code(event.get("rain_level", ""))
                row["event_selection_score"] = float(event.get("event_selection_score", 0.0))
                row["event_historical_trigger_probability"] = float(event.get("historical_trigger_probability", 0.5))
                row["event_historical_trigger_count"] = float(event.get("historical_trigger_count", 0.0))
            rows.append(row)

    return pd.DataFrame(rows, columns=feature_cols)


def normalize_forecast_context(train_context, val_context, test_context, feature_cols=None, issue_age_cap_days=15.0):
    if train_context is None:
        return (None, None, None)
    else:
        feature_cols = list(feature_cols or [f"forecast_{i}" for i in range(train_context.shape[1])])
        train = train_context.astype((np.float32), copy=True)
        val = val_context.astype((np.float32), copy=True)
        test = test_context.astype((np.float32), copy=True)
        for idx, name in enumerate(feature_cols):
            if name in ('forecast_available', 'forecast_is_fallback', 'event_available') or name.startswith("forecast_event_") or name.startswith("event_active_") or name == "event_recent_30d":
                for arr in (train, val, test):
                    arr[:, idx] = np.clip(arr[:, idx], 0.0, 1.0)

                continue
            if name == "forecast_issue_age_days":
                cap = float(issue_age_cap_days)
                if cap <= 0.0:
                    cap = 15.0
                for arr in (train, val, test):
                    arr[:, idx] = np.clip(arr[:, idx], 0.0, cap) / cap

            else:
                if name == "event_age_days":
                    cap = 60.0
                    for arr in (train, val, test):
                        arr[:, idx] = np.clip(arr[:, idx], 0.0, cap) / cap

                    continue
                    values = np.log1p(np.maximum(train[:, idx], 0.0))
                    mean = np.float32(np.mean(values, dtype=(np.float64)))
                    std = np.float32(np.std(values, dtype=(np.float64)))
                    if std < 1e-06:
                        std = np.float32(1.0)
                    for arr in (train, val, test):
                        transformed = (np.log1p(np.maximum(arr[:, idx], 0.0)) - mean) / std
                        arr[:, idx] = np.clip(transformed, -5.0, 5.0)

        return (
         train, val, test)


def target_stage_for_windows(stage_seq, starts, n_his, n_pred):
    target_pos = starts + n_his + n_pred - 1
    return stage_seq[target_pos]


def split_stratified_stage(starts, target_stage, seed=7):
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for stage in np.unique(target_stage):
        group = starts[target_stage == stage].copy()
        rng.shuffle(group)
        n = len(group)
        if n < 6:
            train.extend(group.tolist())
            continue
            n_train = max(1, int(round(0.7 * n)))
            n_val = max(1, int(round(0.15 * n)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            train.extend(group[:n_train].tolist())
            val.extend(group[n_train:n_train + n_val].tolist())
            test.extend(group[n_train + n_val:].tolist())

    return (
     np.array(sorted(train)), np.array(sorted(val)), np.array(sorted(test)))


def split_rolling_cpd(starts, change_points, n_his, n_pred, val_len=18, test_len=18, gap=0):
    n_frame = n_his + n_pred
    folds = []
    for cp in change_points:
        test_start = max(0, cp - n_his)
        test_end = test_start + test_len
        val_end = max(0, test_start - gap)
        val_start = max(0, val_end - val_len)
        train_end = val_start
        train = starts[starts < train_end]
        val = starts[(starts >= val_start) & (starts < val_end)]
        test = starts[(starts >= test_start) & (starts < test_end)]
        if len(train) >= 20 and len(val) >= 4 and len(test) >= 4:
            folds.append((f"cp_{cp}", train, val, test))

    return folds


def rebalance_validation_by_source(folds, forecast_context, forecast_feature_cols, source_col, n_his, n_pred):
    if forecast_context is None or not forecast_feature_cols or source_col not in forecast_feature_cols:
        print(f">> source_balanced_val disabled: missing forecast source column {source_col!r}.")
        return (
         folds, {})
    else:
        source_idx = forecast_feature_cols.index(source_col)
        source_values = np.clip(forecast_context[:, source_idx], 0.0, 1.0) > 0.0
        adjusted = []
        diagnostics = {}
        for fold_name, train_idx, val_idx, test_idx in folds:
            candidates = np.asarray((sorted(set(train_idx.tolist() + val_idx.tolist()))), dtype=(np.int32))
            val_len = len(val_idx)
            if val_len == 0 or len(candidates) < val_len:
                adjusted.append((fold_name, train_idx, val_idx, test_idx))
                continue
                test_fallback = int(np.sum(source_values[test_idx]))
                target_fallback = int(round(val_len * test_fallback / max(len(test_idx), 1)))
                target_fallback = max(0, min(val_len, target_fallback))
                target_official = val_len - target_fallback
                selected = []
                for source_value, target_count in ((True, target_fallback), (False, target_official)):
                    pool = candidates[source_values[candidates] == source_value]
                    if len(pool) > 0 and target_count > 0:
                        selected.extend(pool[-target_count:].tolist())

                if len(selected) < val_len:
                    selected_set = set(selected)
                    fill_pool = [idx for idx in candidates.tolist() if idx not in selected_set]
                    selected.extend(fill_pool[-(val_len - len(selected)):])
                new_val = np.asarray((sorted(selected)), dtype=(np.int32))
                val_set = set(new_val.tolist())
                test_set = set(test_idx.tolist())
                new_train = np.asarray([idx for idx in candidates.tolist() if idx not in val_set if idx not in test_set], dtype=(np.int32))
                adjusted.append((fold_name, new_train, new_val, test_idx))
                diagnostics[fold_name] = {'source_balanced_val':1, 
                 'val_fallback_samples':int(np.sum(source_values[new_val])), 
                 'val_official_samples':int(len(new_val) - np.sum(source_values[new_val])), 
                 'test_fallback_samples':int(test_fallback), 
                 'test_official_samples':int(len(test_idx) - test_fallback)}

        return (
         adjusted, diagnostics)


def evaluate_metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return (rmse, mae)


def evaluate_node_subset(y_true, y_pred, node_mask):
    if node_mask is None or not np.any(node_mask):
        return (float("nan"), float("nan"))
    else:
        return evaluate_metrics(y_true[:, :, node_mask, :], y_pred[:, :, node_mask, :])


def evaluate_sample_subset(y_true, y_pred, sample_mask):
    if sample_mask is None or not np.any(sample_mask):
        return (float("nan"), float("nan"), 0)
    else:
        return (
         *evaluate_metrics(y_true[sample_mask], y_pred[sample_mask]), int(np.sum(sample_mask)))


def horizon_mae_rows(y_true, y_pred, rain_sensitive_mask=None, rain_top_mask=None):
    rows = {}
    n_pred = y_true.shape[1]
    for h in range(n_pred):
        prefix = f"h{h + 1}"
        rows[f"{prefix}_mae"] = evaluate_metrics(y_true[:, h:h + 1], y_pred[:, h:h + 1])[1]
        if rain_sensitive_mask is not None:
            if np.any(rain_sensitive_mask):
                rows[f"{prefix}_rain_sensitive_mae"] = evaluate_metrics(y_true[:, h:h + 1, rain_sensitive_mask, :], y_pred[:, h:h + 1, rain_sensitive_mask, :])[1]
            else:
                rows[f"{prefix}_rain_sensitive_mae"] = float("nan")
            if rain_top_mask is not None and np.any(rain_top_mask):
                rows[f"{prefix}_rain_top_mae"] = evaluate_metrics(y_true[:, h:h + 1, rain_top_mask, :], y_pred[:, h:h + 1, rain_top_mask, :])[1]
            else:
                rows[f"{prefix}_rain_top_mae"] = float("nan")

    return rows


def event_window_mask(aligned_windows, level, lag_days):
    if aligned_windows is None:
        return
    else:
        col = f"rain_event_{level}_{lag_days}d"
        mask = []
        for window in aligned_windows:
            if col not in window.columns:
                mask.append(False)
            else:
                values = window[col].to_numpy(dtype=(np.float32))
                mask.append(bool(np.nanmax(values) > 0.0))

        return np.array(mask, dtype=bool)


def feature_indices(feature_cols, names):
    if feature_cols is None:
        return []
    else:
        col_to_idx = {str(col): i for i, col in enumerate(feature_cols)}
        return [col_to_idx[name] for name in names if name in col_to_idx]


def run_tensor_inference(sess, tensor, x_data, stage_data, batch_size, rain_data=None, rain_tensor=None, forecast_data=None, forecast_tensor=None):
    values = []
    for start in range(0, len(x_data), batch_size):
        end = min(start + batch_size, len(x_data))
        feed = {'data_input:0':x_data[start:end, :N_HIS], 
         'stage_input:0':stage_data[start:end], 
         'keep_prob:0':1.0}
        if rain_data is not None:
            if rain_tensor is not None:
                feed[rain_tensor] = rain_data[start:end]
        if forecast_data is not None:
            if forecast_tensor is not None:
                feed[forecast_tensor] = forecast_data[start:end]
            values.append(sess.run(tensor, feed_dict=feed))

    return np.concatenate(values, axis=0)


def save_fold_prediction_artifacts(output_dir, fold_name, train_idx, val_idx, test_idx, val_true_real, val_pred_real, val_pred_raw_real, val_pred_cal_real, val_base_real, test_true_real, test_pred_real, test_pred_raw_real, test_pred_cal_real, test_base_real, train_true_real=None, train_pred_real=None, train_pred_raw_real=None, train_pred_cal_real=None, train_base_real=None, train_rain=None, test_rain=None, rain_feature_cols=None, train_forecast_context=None, test_forecast_context=None, forecast_feature_cols=None, test_aligned_windows=None, prediction_mode=None):
    """Persist per-sample arrays for post-hoc mechanism MoE analysis."""
    pred_dir = artifacts.prediction_dir(output_dir, fold_name)
    os.makedirs((str(pred_dir)), exist_ok=True)
    np.save(pred_dir / artifacts.TRAIN_INDICES, np.asarray(train_idx, dtype=(np.int32)))
    np.save(pred_dir / artifacts.VAL_INDICES, np.asarray(val_idx, dtype=(np.int32)))
    np.save(pred_dir / artifacts.TEST_INDICES, np.asarray(test_idx, dtype=(np.int32)))
    if train_true_real is not None:
        np.save(pred_dir / artifacts.TRAIN_TRUE_REAL, train_true_real.astype(np.float32))
    if train_pred_real is not None:
        np.save(pred_dir / artifacts.TRAIN_PRED_REAL, train_pred_real.astype(np.float32))
    if train_pred_raw_real is not None:
        np.save(pred_dir / artifacts.TRAIN_PRED_RAW_REAL, train_pred_raw_real.astype(np.float32))
    if train_pred_cal_real is not None:
        np.save(pred_dir / artifacts.TRAIN_PRED_CAL_REAL, train_pred_cal_real.astype(np.float32))
    if train_base_real is not None:
        np.save(pred_dir / artifacts.TRAIN_PERSISTENCE_REAL, train_base_real.astype(np.float32))
    np.save(pred_dir / artifacts.VAL_TRUE_REAL, val_true_real.astype(np.float32))
    np.save(pred_dir / artifacts.VAL_PRED_REAL, val_pred_real.astype(np.float32))
    np.save(pred_dir / artifacts.VAL_PRED_RAW_REAL, val_pred_raw_real.astype(np.float32))
    np.save(pred_dir / artifacts.VAL_PRED_CAL_REAL, val_pred_cal_real.astype(np.float32))
    np.save(pred_dir / artifacts.VAL_PERSISTENCE_REAL, val_base_real.astype(np.float32))
    np.save(pred_dir / artifacts.TEST_TRUE_REAL, test_true_real.astype(np.float32))
    np.save(pred_dir / artifacts.TEST_PRED_REAL, test_pred_real.astype(np.float32))
    np.save(pred_dir / artifacts.TEST_PRED_RAW_REAL, test_pred_raw_real.astype(np.float32))
    np.save(pred_dir / artifacts.TEST_PRED_CAL_REAL, test_pred_cal_real.astype(np.float32))
    np.save(pred_dir / artifacts.TEST_PERSISTENCE_REAL, test_base_real.astype(np.float32))
    if train_rain is not None:
        np.save(pred_dir / artifacts.TRAIN_RAIN, train_rain.astype(np.float32))
    if test_rain is not None:
        np.save(pred_dir / artifacts.TEST_RAIN, test_rain.astype(np.float32))
    if train_forecast_context is not None:
        np.save(pred_dir / artifacts.TRAIN_FORECAST_CONTEXT, train_forecast_context.astype(np.float32))
    if test_forecast_context is not None:
        np.save(pred_dir / artifacts.TEST_FORECAST_CONTEXT, test_forecast_context.astype(np.float32))
    if test_aligned_windows is not None:
        aligned_path = pred_dir / artifacts.TEST_ALIGNED_RAIN_WINDOWS
        aligned_rows = []
        for sample_idx, window in enumerate(test_aligned_windows):
            for window_pos, row in enumerate(window.to_dict(orient="records")):
                row = dict(row)
                row["sample_index"] = sample_idx
                row["window_pos"] = window_pos
                aligned_rows.append(row)

        if aligned_rows:
            fieldnames = [
             "sample_index", "window_pos"] + [key for key in aligned_rows[0].keys() if key not in ('sample_index',
                                                                                                   'window_pos')]
            with open((str(aligned_path)), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(aligned_rows)
    if rain_feature_cols is not None:
        with open((pred_dir / artifacts.RAIN_FEATURE_COLS), "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, rain_feature_cols)))
    if forecast_feature_cols is not None:
        with open((pred_dir / artifacts.FORECAST_FEATURE_COLS), "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, forecast_feature_cols)))
    with open((pred_dir / artifacts.METADATA), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'prediction_mode', 'train_n', 'val_n', 'test_n'])
        writer.writeheader()
        writer.writerow({'fold':fold_name, 
         'prediction_mode':prediction_mode, 
         'train_n':len(train_idx), 
         'val_n':len(val_idx), 
         'test_n':len(test_idx)})
    return str(pred_dir)


def rain_susceptibility_masks(rain_susceptibility, top_quantile):
    if rain_susceptibility is None:
        return (None, None)
    else:
        susceptibility = np.asarray(rain_susceptibility, dtype=(np.float32))
        active_mask = susceptibility > 0.0
        if not np.any(active_mask):
            return (
             active_mask, active_mask)
        threshold = float(np.quantile(susceptibility[active_mask], top_quantile))
        top_mask = active_mask & (susceptibility >= threshold)
        return (active_mask, top_mask)


def percentage_gain(reference_mae, model_mae):
    if not np.isfinite(reference_mae) or not np.isfinite(model_mae):
        return float("nan")
    else:
        return 100.0 * (reference_mae - model_mae) / max(reference_mae, 1e-06)


def inverse_transform(data, mean, std):
    return data * std + mean


def load_train_distillation_targets(targets_dir, train_idx, positive_only=False, gain_col='teacher_gain_rain_sensitive_mae', min_mean_gain=None, sample_filter_col=None, sample_weight_col=None, sample_weight_scale=1.0, horizon_start=0, n_pred=5):
    if not targets_dir:
        return (None, None)
    else:
        teacher_path = os.path.join(targets_dir, "train_teacher_pred_real.npy")
        metrics_path = os.path.join(targets_dir, "distillation_sample_metrics.csv")
        if not os.path.exists(teacher_path):
            raise FileNotFoundError(f"Missing distillation teacher file: {teacher_path}")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Missing distillation metrics file: {metrics_path}")
        teacher = np.load(teacher_path).astype(np.float32)
        with open(metrics_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        starts = np.asarray([int(float(row["sample_start"])) for row in rows], dtype=(np.int32))
        expected = np.asarray(train_idx, dtype=(np.int32))
        if teacher.shape[0] != len(expected):
            raise ValueError(f"Distillation teacher samples={teacher.shape[0]} != train samples={len(expected)}")
        if starts.shape[0] != len(expected) or not np.array_equal(starts, expected):
            raise ValueError("Distillation sample_start values do not match the current train split.")
        gains = None
        if gain_col in rows[0]:
            gains = np.asarray([float(row[gain_col]) for row in rows], dtype=(np.float32))
        if min_mean_gain is not None:
            if gains is None:
                raise ValueError(f"Distillation metrics do not contain gain column: {gain_col}")
            mean_gain = float(np.mean(gains))
            if mean_gain < float(min_mean_gain):
                print(f">> distillation disabled: mean {gain_col}={mean_gain:.6f} < min_mean_gain={float(min_mean_gain):.6f}")
                return (None, None)
        weight_path = os.path.join(targets_dir, "train_distill_weights.npy")
        if os.environ.get("IGNORE_DISTILL_WEIGHTS", "0") == "1":
            print(">> IGNORE_DISTILL_WEIGHTS is set, skipping 3D weights")
        elif os.path.exists(weight_path):
            sample_weight = np.load(weight_path).astype(np.float32)
            if int(horizon_start) > 0:
                horizon_mask = np.zeros((1, int(n_pred), 1, 1), dtype=(np.float32))
                start = max(0, min(int(horizon_start) - 1, int(n_pred) - 1))
                horizon_mask[:, start:, :, :] = 1.0
                sample_weight = sample_weight * horizon_mask
            return (teacher, sample_weight)

        sample_weight = None
        if positive_only:
            if gain_col not in rows[0]:
                raise ValueError(f"Distillation metrics do not contain gain column: {gain_col}")
            sample_weight = (gains > 0.0).astype(np.float32)
            if not np.any(sample_weight > 0.0):
                raise ValueError(f"No positive-gain samples found for distillation column: {gain_col}")
        if sample_filter_col:
            if sample_filter_col not in rows[0]:
                raise ValueError(f"Distillation metrics do not contain filter column: {sample_filter_col}")
            filter_weight = np.asarray([float(row.get(sample_filter_col, 0.0)) for row in rows], dtype=(np.float32))
            filter_weight = (filter_weight > 0.0).astype(np.float32)
            sample_weight = filter_weight if sample_weight is None else sample_weight * filter_weight
        if sample_weight_col:
            if sample_weight_col not in rows[0]:
                raise ValueError(f"Distillation metrics do not contain weight column: {sample_weight_col}")
            extra = np.asarray([float(row.get(sample_weight_col, 0.0)) for row in rows], dtype=(np.float32))
            extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
            extra = 1.0 + float(sample_weight_scale) * np.maximum(extra, 0.0)
            sample_weight = extra if sample_weight is None else sample_weight * extra
        if sample_weight is not None:
            if not np.any(sample_weight > 0.0):
                raise ValueError("No positive-weight samples found for distillation.")
            sample_weight = sample_weight.astype(np.float32).reshape(-1, 1, 1, 1)
            if int(horizon_start) > 0:
                horizon_mask = np.zeros((1, int(n_pred), 1, 1), dtype=(np.float32))
                start = max(0, min(int(horizon_start) - 1, int(n_pred) - 1))
                horizon_mask[:, start:, :, :] = 1.0
                sample_weight = sample_weight * horizon_mask
        return (
         teacher, sample_weight)


def calibrate_node_gain(val_pred, val_true, val_base, shrink):
    pred_delta = val_pred - val_base
    true_delta = val_true - val_base
    numerator = np.sum((pred_delta * true_delta), axis=(0, 1, 3), keepdims=True)
    denominator = np.sum((np.square(pred_delta)), axis=(0, 1, 3), keepdims=True) + 1e-06
    return shrink * np.clip(numerator / denominator, 0.0, 1.0)


def run_inference(sess, pred_tensor, x_data, stage_data, batch_size, rain_data=None, rain_tensor=None, forecast_data=None, forecast_tensor=None):
    preds = []
    for start in range(0, len(x_data), batch_size):
        end = min(start + batch_size, len(x_data))
        feed = {'data_input:0':x_data[start:end, :N_HIS], 
         'stage_input:0':stage_data[start:end], 
         'keep_prob:0':1.0}
        if rain_data is not None:
            if rain_tensor is not None:
                feed[rain_tensor] = rain_data[start:end]
        if forecast_data is not None:
            if forecast_tensor is not None:
                feed[forecast_tensor] = forecast_data[start:end]
            pred = sess.run(pred_tensor, feed_dict=feed)
            preds.append(pred)

    return np.concatenate(preds, axis=0)


def stage_histogram(stage_seq, starts, n_his, n_pred):
    target_stage = target_stage_for_windows(stage_seq, starts, n_his, n_pred)
    unique, counts = np.unique(target_stage, return_counts=True)
    return ";".join(f"{int(k)}:{int(v)}" for k, v in zip(unique, counts))


def train_one_fold(fold_name, raw_seq, stage_seq, adj, train_idx, val_idx, test_idx, args, rain_seq=None, rain_feature_cols=None, rain_susceptibility=None, hydro_flow_weight=None, aligned_rain=None, forecast_context=None, forecast_feature_cols=None, event_context=None, event_feature_cols=None, split_diagnostics=None):
    from pathlib import Path
    pred_dir = Path(args.output_dir) / fold_name / "predictions"
    if (pred_dir / "test_pred_real.npy").exists():
        print(f"Skipping fold {fold_name} as predictions already exist in {pred_dir}")
        return None
    tf.compat.v1.reset_default_graph()
    fold_seed = args.seed + sum(ord(ch) for ch in fold_name) % 10000
    np.random.seed(fold_seed)
    tf.compat.v1.set_random_seed(fold_seed)
    tf.compat.v1.add_to_collection("graph_kernel", tf.constant(adj, dtype=(tf.float32)))
    rain_sensitive_mask, rain_top_mask = rain_susceptibility_masks(rain_susceptibility, args.rain_sensitive_quantile)
    train_values = []
    n_frame = args.n_his + args.n_pred
    for start in train_idx:
        train_values.append(raw_seq[start:start + n_frame])

    train_values = np.concatenate(train_values, axis=0)
    mean = np.mean(train_values, dtype=(np.float64)).astype(np.float32)
    std = np.std(train_values, dtype=(np.float64)).astype(np.float32)
    std = np.maximum(std, np.float32(1e-06))
    norm_seq = ((raw_seq - mean) / std).astype(np.float32)
    train_x, train_s = make_windows(norm_seq, stage_seq, train_idx, args.n_his, args.n_pred)
    val_x, val_s = make_windows(norm_seq, stage_seq, val_idx, args.n_his, args.n_pred)
    test_x, test_s = make_windows(norm_seq, stage_seq, test_idx, args.n_his, args.n_pred)
    train_rain = make_rain_windows(rain_seq, train_idx, args.n_his)
    val_rain = make_rain_windows(rain_seq, val_idx, args.n_his)
    test_rain = make_rain_windows(rain_seq, test_idx, args.n_his)
    train_teacher = None
    train_teacher_weight = None
    if args.distill_targets_dir:
        if args.distill_weight > 0.0:
            gain_col = "teacher_gain_global_mae" if args.distill_scope == "all" else "teacher_gain_rain_sensitive_mae"
            train_teacher_real, train_teacher_weight = load_train_distillation_targets((args.distill_targets_dir),
              train_idx,
              positive_only=(args.distill_positive_only),
              gain_col=gain_col,
              min_mean_gain=(args.distill_min_mean_gain),
              sample_filter_col=(args.distill_sample_filter_col),
              sample_weight_col=(args.distill_sample_weight_col),
              sample_weight_scale=(args.distill_sample_weight_scale),
              horizon_start=(args.distill_horizon_start),
              n_pred=(args.n_pred))
            if train_teacher_real is not None:
                expected_shape = train_x[:, args.n_his:args.n_his + args.n_pred].shape
                if train_teacher_real.shape != expected_shape:
                    raise ValueError(f"Distillation teacher shape {train_teacher_real.shape} != train target shape {expected_shape}")
                train_teacher = ((train_teacher_real - mean) / std).astype(np.float32)
    test_aligned_windows = make_aligned_windows(aligned_rain, test_idx, args.n_his)
    train_forecast_raw = select_forecast_context(forecast_context, train_idx)
    val_forecast_raw = select_forecast_context(forecast_context, val_idx)
    test_forecast_raw = select_forecast_context(forecast_context, test_idx)
    train_forecast, val_forecast, test_forecast = normalize_forecast_context(train_forecast_raw,
      val_forecast_raw,
      test_forecast_raw,
      feature_cols=forecast_feature_cols)
    event_sample_weight = build_event_sample_weights(event_context,
      event_feature_cols,
      strength=(args.event_loss_weight),
      mode=(args.event_loss_weight_mode))
    train_event_weight = select_forecast_context(event_sample_weight, train_idx)
    val_event_weight = select_forecast_context(event_sample_weight, val_idx)
    x = tf.compat.v1.placeholder((tf.float32), [None, args.n_his, raw_seq.shape[1], 1], name="data_input")
    y_true = tf.compat.v1.placeholder((tf.float32), [None, args.n_pred, raw_seq.shape[1], 1], name="data_label")
    y_teacher = None
    y_teacher_weight = None
    if train_teacher is not None:
        y_teacher = tf.compat.v1.placeholder((tf.float32),
          [
         None, args.n_pred, raw_seq.shape[1], 1],
          name="distill_teacher")
        if train_teacher_weight is not None:
            y_teacher_weight = tf.compat.v1.placeholder((tf.float32),
              shape=[None, int(args.n_pred), None, 1],
              name="distill_teacher_weight")
    stage_input = tf.compat.v1.placeholder((tf.int32), [None, args.n_his], name="stage_input")
    rain_input = None
    if train_rain is not None:
        rain_input = tf.compat.v1.placeholder((tf.float32), [None, args.n_his, train_rain.shape[2]], name="rain_input")
    forecast_input = None
    if train_forecast is not None:
        forecast_input = tf.compat.v1.placeholder((tf.float32),
          [
         None, train_forecast.shape[1]],
          name="forecast_context")
    event_weight_input = None
    if train_event_weight is not None:
        event_weight_input = tf.compat.v1.placeholder((tf.float32),
          [
         None, 1, 1, 1],
          name="event_sample_weight")
    else:
        forecast_source_index = None
        use_source_aware_residual = bool(args.source_aware_residual and forecast_input is not None)
        if use_source_aware_residual:
            if forecast_feature_cols and args.forecast_source_col in forecast_feature_cols:
                forecast_source_index = forecast_feature_cols.index(args.forecast_source_col)
            else:
                use_source_aware_residual = False
                print(f">> source_aware_residual disabled: missing forecast source column {args.forecast_source_col!r}.")
    rain_susceptibility_tensor = None
    use_dynamic_rain_graph = bool(args.dynamic_rain_graph and rain_input is not None and rain_susceptibility is not None)
    if use_dynamic_rain_graph:
        rain_susceptibility_tensor = tf.constant(rain_susceptibility, dtype=(tf.float32), name="rain_susceptibility")
    hydro_flow_weight_tensor = None
    if use_dynamic_rain_graph:
        if args.hydro_flow_gate:
            if hydro_flow_weight is not None:
                hydro_flow_weight_tensor = tf.constant(hydro_flow_weight, dtype=(tf.float32), name="hydro_flow_weight")
    keep_prob = tf.compat.v1.placeholder((tf.float32), name="keep_prob")
    lr_placeholder = tf.compat.v1.placeholder((tf.float32), name="dynamic_lr")
    y_pred, copy_loss, model_aux = build_model(x,
      stage_input,
      (args.n_his),
      3,
      3,
      [
     [
      1, 8, 16], [16, 8, 16]],
      keep_prob,
      n_pred=(args.n_pred),
      residual_scale=(args.residual_scale),
      rain_features=rain_input,
      rain_susceptibility=rain_susceptibility_tensor,
      hydro_flow_weight=hydro_flow_weight_tensor,
      dynamic_rain_graph=use_dynamic_rain_graph,
      temporal_encoder=(args.temporal_encoder),
      forecast_context=forecast_input,
      source_aware_residual=use_source_aware_residual,
      forecast_source_index=forecast_source_index,
      source_delta_scale=(args.source_delta_scale),
      return_aux=True)
    stage_float = tf.cast(stage_input, tf.float32)
    stage_span = tf.reduce_max(stage_float, axis=1) - tf.reduce_min(stage_float, axis=1)
    has_cpd_transition = tf.cast(stage_span > 0.0, tf.float32)
    stage_level = tf.reduce_mean(stage_float, axis=1) / 5.0
    sample_weight = tf.reshape(1.0 + args.cpd_transition_weight * has_cpd_transition + args.cpd_stage_weight * stage_level, [
     -1, 1, 1, 1])
    if rain_input is not None:
        rain_intensity = tf.reshape(tf.reduce_mean((tf.nn.relu(rain_input)), axis=[1, 2]), [-1, 1, 1, 1])
        sample_weight += args.rain_weight * tf.minimum(rain_intensity, 3.0)
    if event_weight_input is not None:
        sample_weight *= event_weight_input
    node_loss_weight = 1.0
    if rain_susceptibility is not None:
        if args.rain_node_loss_weight > 0.0:
            rain_node_weight = np.asarray(rain_susceptibility, dtype=(np.float32))
            if np.max(rain_node_weight) > 0.0:
                rain_node_weight = rain_node_weight / np.max(rain_node_weight)
            node_loss_weight = tf.constant((1.0 + args.rain_node_loss_weight * rain_node_weight.reshape(1, 1, -1, 1)),
              dtype=(tf.float32),
              name="rain_node_loss_weight")
    loss_weight = sample_weight * node_loss_weight
    distill_loss = 0.0
    if y_teacher is not None:
        if getattr(args, 'distill_residual_only', False) and "source_delta" in model_aux and "baseline_pred" in model_aux:
            teacher_delta = y_teacher - tf.stop_gradient(model_aux["baseline_pred"])
            student_delta = args.residual_scale * args.source_delta_scale * model_aux["source_delta"]
            distill_error = tf.abs(student_delta - teacher_delta)
        else:
            distill_error = tf.abs(y_pred - y_teacher)
        distill_quadratic = tf.minimum(distill_error, 1.0)
        distill_huber = 0.5 * tf.square(distill_quadratic) + (distill_error - distill_quadratic)
        if args.distill_scope == "rain_sensitive":
            if rain_susceptibility is not None:
                distill_mask = np.asarray((rain_susceptibility > 0.0), dtype=(np.float32)).reshape(1, 1, -1, 1)
                distill_huber = distill_huber * tf.constant(distill_mask, dtype=(tf.float32))

        if getattr(args, 'distill_learnable_mask', False):
            num_nodes = raw_seq.shape[1]
            learnable_mask = tf.compat.v1.get_variable("distill_learnable_mask", shape=[1, 1, num_nodes, 1], initializer=tf.constant_initializer(0.0))
            if args.distill_scope == "rain_sensitive" and rain_susceptibility is not None:
                # -1e9 for non-rain sensitive nodes to ignore them in softmax
                mask_logits = learnable_mask + (distill_mask - 1.0) * 1e9
                num_rain_nodes = tf.reduce_sum(distill_mask)
                spatial_mask = tf.nn.softmax(mask_logits, axis=2) * tf.maximum(num_rain_nodes, 1.0)
                distill_huber = distill_huber * spatial_mask
            else:
                spatial_mask = tf.nn.softmax(learnable_mask, axis=2) * float(num_nodes)
                distill_huber = distill_huber * spatial_mask

        if y_teacher_weight is not None:
            # Non-active period down-weighting: stage_level is 0.0 to 1.0 based on CPD stages
            distill_stage_multiplier = tf.maximum(0.1, tf.reshape(stage_level, [-1, 1, 1, 1]))
            weighted = distill_huber * y_teacher_weight * distill_stage_multiplier
            # Normalizing only by y_teacher_weight to allow the batch loss to drop when stage is inactive
            distill_loss = tf.reduce_mean(weighted) / tf.maximum(tf.reduce_mean(y_teacher_weight), 1e-06)
        else:
            distill_stage_multiplier = tf.maximum(0.1, tf.reshape(stage_level, [-1, 1, 1, 1]))
            distill_loss = tf.reduce_mean(distill_huber * distill_stage_multiplier)
    last_step = x[:, -1:, :, :]
    diff_pred = y_pred - last_step
    diff_true = y_true - last_step
    residual_error = diff_pred - diff_true
    abs_error = tf.abs(residual_error)
    quadratic = tf.minimum(abs_error, 1.0)
    huber = 0.5 * tf.square(quadratic) + (abs_error - quadratic)
    motion_weight = 1.0 + 2.0 * tf.minimum(tf.abs(diff_true), 2.0)
    horizon_loss_weight = 1.0
    if args.horizon_loss_gamma > 0.0:
        if args.n_pred > 1:
            horizon_axis = np.linspace(0.0, 1.0, (args.n_pred), dtype=(np.float32)).reshape(1, args.n_pred, 1, 1)
            horizon_loss_weight = tf.constant((1.0 + np.float32(args.horizon_loss_gamma) * horizon_axis),
              dtype=(tf.float32),
              name="horizon_loss_weight")
            loss_weight = loss_weight * horizon_loss_weight
    hydro_aux_loss = 0.0
    hydro_attention_loss = 0.0
    if rain_input is not None:
        if args.hydro_aux_weight > 0.0:
            if rain_susceptibility is not None:
                event_cols = feature_indices(rain_feature_cols, [
                 'rain_event_p90_3d', 'rain_event_p90_7d', 
                 'rain_event_p90_15d', 
                 'rain_event_p95_3d', 
                 'rain_event_p95_7d', 'rain_event_p95_15d'])
                if event_cols:
                    event_signal = tf.reduce_max(tf.gather(rain_input, event_cols, axis=2), axis=[1, 2])
                    event_signal = tf.reshape(tf.clip_by_value(event_signal, 0.0, 1.0), [-1, 1, 1, 1])
                    rain_node_mask = np.asarray((rain_susceptibility > 0.0), dtype=(np.float32)).reshape(1, 1, -1, 1)
                    hydro_aux_loss = tf.reduce_mean(event_signal * tf.constant(rain_node_mask, dtype=(tf.float32)) * huber)
    temporal_attention = model_aux.get("temporal_attention")
    if rain_input is not None:
        if temporal_attention is not None:
            if args.hydro_attention_weight > 0.0:
                memory_cols = feature_indices(rain_feature_cols, ["hydro_memory_index"])
                if memory_cols:
                    memory = tf.gather(rain_input, (memory_cols[0]), axis=2)
                    hydro_attention_loss = tf.reduce_mean(temporal_attention * tf.nn.relu(-memory))
    temporal_smoothness_loss = 0.0
    if getattr(args, 'temporal_smoothness_weight', 0.0) > 0.0 and args.n_pred > 1:
        temporal_diff = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        temporal_smoothness_loss = tf.reduce_mean(tf.square(temporal_diff))
    base_loss = tf.reduce_mean(loss_weight * huber) + 0.2 * tf.reduce_mean(loss_weight * tf.square(y_pred - y_true)) + 0.5 * tf.reduce_mean(loss_weight * motion_weight * tf.abs(residual_error)) + 0.2 * tf.reduce_mean(loss_weight * tf.abs(y_pred - y_true)) + 0.05 * tf.reduce_mean(loss_weight * tf.nn.relu(-(diff_pred * diff_true))) + args.hydro_aux_weight * hydro_aux_loss + args.hydro_attention_weight * hydro_attention_loss + getattr(args, 'temporal_smoothness_weight', 0.0) * temporal_smoothness_loss + 0.5 * copy_loss
    distill_weight_ph = tf.compat.v1.placeholder_with_default(float(args.distill_weight), shape=(), name="distill_weight_ph")
    train_loss = base_loss + distill_weight_ph * distill_loss
    train_op = tf.compat.v1.train.AdamOptimizer(lr_placeholder).minimize(train_loss)
    saver = tf.compat.v1.train.Saver(max_to_keep=1)
    model_dir = os.path.join(args.output_dir, fold_name, "models")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    best_val = float("inf")
    wait = 0
    num_batch = int(np.ceil(len(train_x) / args.batch_size))
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for epoch in range(args.epochs):
            if args.distill_weight_end is not None and args.distill_anneal_epoch > 0:
                progress = min(1.0, epoch / args.distill_anneal_epoch)
                distill_weight_val = args.distill_weight - progress * (args.distill_weight - args.distill_weight_end)
            else:
                distill_weight_val = args.distill_weight
            order = np.random.permutation(len(train_x))
            losses = []
            t0 = time.time()
            for batch in range(num_batch):
                batch_idx = order[batch * args.batch_size:(batch + 1) * args.batch_size]
                batch_stage = train_s[batch_idx]
                stage_mean = np.mean(batch_stage)
                lr_scale = 0.8 if stage_mean < 1.5 else 1.0 if stage_mean < 2.5 else 1.2
                lr = args.lr * 0.8 ** (epoch // 10) * lr_scale
                feed = {x: (train_x[batch_idx, :args.n_his]), 
                 y_true: (train_x[batch_idx, args.n_his:args.n_his + args.n_pred]), 
                 stage_input: batch_stage, 
                 keep_prob: 0.8, 
                 lr_placeholder: lr,
                 distill_weight_ph: distill_weight_val}
                if rain_input is not None:
                    feed[rain_input] = train_rain[batch_idx]
                if forecast_input is not None:
                    feed[forecast_input] = train_forecast[batch_idx]
                if event_weight_input is not None:
                    feed[event_weight_input] = train_event_weight[batch_idx]
                if y_teacher is not None:
                    feed[y_teacher] = train_teacher[batch_idx]
                if y_teacher_weight is not None:
                    feed[y_teacher_weight] = train_teacher_weight[batch_idx]
                _, loss_val = sess.run([train_op, train_loss], feed_dict=feed)
                losses.append(float(loss_val))

            val_feed = {x: (val_x[:, :args.n_his]), 
             y_true: (val_x[:, args.n_his:args.n_his + args.n_pred]), 
             stage_input: val_s, 
             keep_prob: 1.0, 
             lr_placeholder: (args.lr)}
            if rain_input is not None:
                val_feed[rain_input] = val_rain
            if forecast_input is not None:
                val_feed[forecast_input] = val_forecast
            if event_weight_input is not None:
                val_feed[event_weight_input] = val_event_weight
            val_loss = float(sess.run(base_loss, feed_dict=val_feed))
            selection_value = val_loss
            selection_suffix = ""
            if args.selection_metric in ('rain_sensitive_mae', 'val_real_mae', 'val_real_rain_sensitive_mae'):
                val_pred_epoch = run_inference(sess, y_pred, val_x, val_s, args.batch_size, rain_data=val_rain, rain_tensor=rain_input, forecast_data=val_forecast, forecast_tensor=forecast_input)
                val_true_epoch = val_x[:, args.n_his:args.n_his + args.n_pred]
                if args.selection_metric == "rain_sensitive_mae":
                    if rain_sensitive_mask is not None and np.any(rain_sensitive_mask):
                        _, val_rain_mae = evaluate_node_subset(val_true_epoch, val_pred_epoch, rain_sensitive_mask)
                        selection_value = val_rain_mae
                        selection_suffix = f" rain_val_mae {val_rain_mae:.4f}"
                else:
                    val_pred_epoch_real = inverse_transform(val_pred_epoch, mean, std)
                    val_true_epoch_real = inverse_transform(val_true_epoch, mean, std)
                    if args.selection_metric == "val_real_mae":
                        _, val_real_mae = evaluate_metrics(val_true_epoch_real, val_pred_epoch_real)
                        selection_value = val_real_mae
                        selection_suffix = f" val_real_mae {val_real_mae:.4f}"
                    elif args.selection_metric == "val_real_rain_sensitive_mae":
                        if rain_sensitive_mask is not None and np.any(rain_sensitive_mask):
                            _, val_real_rain_mae = evaluate_node_subset(val_true_epoch_real, val_pred_epoch_real, rain_sensitive_mask)
                            selection_value = val_real_rain_mae
                            selection_suffix = f" val_real_rain_mae {val_real_rain_mae:.4f}"
            print(f"{fold_name} epoch {epoch + 1:02d}/{args.epochs} train {np.mean(losses):.4f} val {val_loss:.4f}{selection_suffix} time {time.time() - t0:.1f}s")
            if selection_value < best_val:
                best_val = float(selection_value)
                wait = 0
                saver.save(sess, os.path.join(model_dir, "best_model"))
            else:
                wait += 1
                if wait >= args.patience:
                    break

        saver.restore(sess, os.path.join(model_dir, "best_model"))
        val_pred = run_inference(sess,
          y_pred,
          val_x,
          val_s,
          (args.batch_size),
          rain_data=val_rain,
          rain_tensor=rain_input,
          forecast_data=val_forecast,
          forecast_tensor=forecast_input)
        test_pred = run_inference(sess,
          y_pred,
          test_x,
          test_s,
          (args.batch_size),
          rain_data=test_rain,
          rain_tensor=rain_input,
          forecast_data=test_forecast,
          forecast_tensor=forecast_input)
        train_pred = None
        if args.save_fold_predictions:
            train_pred = run_inference(sess,
              y_pred,
              train_x,
              train_s,
              (args.batch_size),
              rain_data=train_rain,
              rain_tensor=rain_input,
              forecast_data=train_forecast,
              forecast_tensor=forecast_input)
        test_attention = None
        if temporal_attention is not None:
            test_attention = run_tensor_inference(sess,
              temporal_attention,
              test_x,
              test_s,
              (args.batch_size),
              rain_data=test_rain,
              rain_tensor=rain_input,
              forecast_data=test_forecast,
              forecast_tensor=forecast_input)
    train_true = train_x[:, args.n_his:args.n_his + args.n_pred]
    val_true = val_x[:, args.n_his:args.n_his + args.n_pred]
    test_true = test_x[:, args.n_his:args.n_his + args.n_pred]
    train_base = np.repeat((train_x[:, args.n_his - 1:args.n_his]), (args.n_pred), axis=1)
    val_base = np.repeat((val_x[:, args.n_his - 1:args.n_his]), (args.n_pred), axis=1)
    test_base = np.repeat((test_x[:, args.n_his - 1:args.n_his]), (args.n_pred), axis=1)
    train_true_real = inverse_transform(train_true, mean, std)
    train_base_real = inverse_transform(train_base, mean, std)
    train_pred_raw_real = inverse_transform(train_pred, mean, std) if train_pred is not None else None
    val_pred_real = inverse_transform(val_pred, mean, std)
    val_true_real = inverse_transform(val_true, mean, std)
    val_base_real = inverse_transform(val_base, mean, std)
    test_pred_raw_real = inverse_transform(test_pred, mean, std)
    test_true_real = inverse_transform(test_true, mean, std)
    test_base_real = inverse_transform(test_base, mean, std)
    node_gain = calibrate_node_gain(val_pred_real, val_true_real, val_base_real, args.node_gain_shrink)
    train_pred_cal_real = train_base_real + node_gain * (train_pred_raw_real - train_base_real) if train_pred_raw_real is not None else None
    val_pred_cal_real = val_base_real + node_gain * (val_pred_real - val_base_real)
    test_pred_cal_real = test_base_real + node_gain * (test_pred_raw_real - test_base_real)
    val_raw_rmse, val_raw_mae = evaluate_metrics(val_true_real, val_pred_real)
    val_cal_rmse, val_cal_mae = evaluate_metrics(val_true_real, val_pred_cal_real)
    raw_score = val_raw_rmse + 0.25 * val_raw_mae
    cal_score = val_cal_rmse + 0.25 * val_cal_mae
    if raw_score <= cal_score:
        prediction_mode = "raw"
        test_pred_real = test_pred_raw_real
        train_pred_real = train_pred_raw_real
    else:
        prediction_mode = "node_calibrated"
        test_pred_real = test_pred_cal_real
        train_pred_real = train_pred_cal_real
    model_rmse, model_mae = evaluate_metrics(test_true_real, test_pred_real)
    base_rmse, base_mae = evaluate_metrics(test_true_real, test_base_real)
    raw_rmse, raw_mae = evaluate_metrics(test_true_real, test_pred_raw_real)
    rain_sensitive_rmse, rain_sensitive_mae = evaluate_node_subset(test_true_real, test_pred_real, rain_sensitive_mask)
    rain_sensitive_base_rmse, rain_sensitive_base_mae = evaluate_node_subset(test_true_real, test_base_real, rain_sensitive_mask)
    rain_top_rmse, rain_top_mae = evaluate_node_subset(test_true_real, test_pred_real, rain_top_mask)
    rain_top_base_rmse, rain_top_base_mae = evaluate_node_subset(test_true_real, test_base_real, rain_top_mask)
    event_metrics = {}
    for level in ('p90', 'p95'):
        for lag_days in (3, 7, 15):
            sample_mask = event_window_mask(test_aligned_windows, level, lag_days)
            event_rmse, event_mae, event_n = evaluate_sample_subset(test_true_real, test_pred_real, sample_mask)
            _, event_base_mae, _ = evaluate_sample_subset(test_true_real, test_base_real, sample_mask)
            prefix = f"{level}_{lag_days}d"
            event_metrics[f"{prefix}_samples"] = event_n
            event_metrics[f"{prefix}_mae"] = event_mae
            event_metrics[f"{prefix}_persistence_mae"] = event_base_mae
            event_metrics[f"{prefix}_mae_gain_pct"] = percentage_gain(event_base_mae, event_mae)

    attention_mean = float("nan")
    attention_peak_lag = -1
    if test_attention is not None:
        attention_dir = os.path.join(args.output_dir, fold_name)
        os.makedirs(attention_dir, exist_ok=True)
        np.save(os.path.join(attention_dir, "hydro_temporal_attention.npy"), test_attention.astype(np.float32))
        if rain_feature_cols is not None:
            with open((os.path.join(attention_dir, "rain_feature_cols.txt")), "w", encoding="utf-8") as f:
                f.write("\n".join(map(str, rain_feature_cols)))
        attention_profile = np.mean(test_attention, axis=0)
        attention_mean = float(np.mean(test_attention))
        attention_peak_lag = int(np.argmax(attention_profile))
    prediction_artifacts_dir = ""
    if args.save_fold_predictions:
        val_pred_selected_real = val_pred_real if raw_score <= cal_score else val_pred_cal_real
        prediction_artifacts_dir = save_fold_prediction_artifacts((args.output_dir),
          fold_name,
          train_idx,
          val_idx,
          test_idx,
          val_true_real,
          val_pred_selected_real,
          val_pred_real,
          val_pred_cal_real,
          val_base_real,
          test_true_real,
          test_pred_real,
          test_pred_raw_real,
          test_pred_cal_real,
          test_base_real,
          train_true_real=(train_true_real if train_pred is not None else None),
          train_pred_real=train_pred_real,
          train_pred_raw_real=train_pred_raw_real,
          train_pred_cal_real=train_pred_cal_real,
          train_base_real=(train_base_real if train_pred is not None else None),
          train_rain=train_rain,
          test_rain=test_rain,
          rain_feature_cols=rain_feature_cols,
          train_forecast_context=train_forecast,
          test_forecast_context=test_forecast,
          forecast_feature_cols=forecast_feature_cols,
          test_aligned_windows=test_aligned_windows,
          prediction_mode=prediction_mode)
    row = {'fold':fold_name, 
     'train_n':len(train_idx), 
     'val_n':len(val_idx), 
     'test_n':len(test_idx), 
     'train_stage':stage_histogram(stage_seq, train_idx, args.n_his, args.n_pred), 
     'val_stage':stage_histogram(stage_seq, val_idx, args.n_his, args.n_pred), 
     'test_stage':stage_histogram(stage_seq, test_idx, args.n_his, args.n_pred), 
     'model_rmse':model_rmse, 
     'model_mae':model_mae, 
     'persistence_rmse':base_rmse, 
     'persistence_mae':base_mae, 
     'raw_rmse':raw_rmse, 
     'raw_mae':raw_mae, 
     'rain_sensitive_nodes':int(np.sum(rain_sensitive_mask)) if rain_sensitive_mask is not None else 0, 
     'rain_sensitive_rmse':rain_sensitive_rmse, 
     'rain_sensitive_mae':rain_sensitive_mae, 
     'rain_sensitive_persistence_rmse':rain_sensitive_base_rmse, 
     'rain_sensitive_persistence_mae':rain_sensitive_base_mae, 
     'rain_sensitive_mae_gain_pct':percentage_gain(rain_sensitive_base_mae, rain_sensitive_mae), 
     'rain_top_nodes':int(np.sum(rain_top_mask)) if rain_top_mask is not None else 0, 
     'rain_top_quantile':args.rain_sensitive_quantile, 
     'rain_top_rmse':rain_top_rmse, 
     'rain_top_mae':rain_top_mae, 
     'rain_top_persistence_rmse':rain_top_base_rmse, 
     'rain_top_persistence_mae':rain_top_base_mae, 
     'rain_top_mae_gain_pct':percentage_gain(rain_top_base_mae, rain_top_mae), 
     'val_raw_rmse':val_raw_rmse, 
     'val_raw_mae':val_raw_mae, 
     'val_cal_rmse':val_cal_rmse, 
     'val_cal_mae':val_cal_mae, 
     'prediction_mode':prediction_mode, 
     'rmse_gain_pct':100.0 * (base_rmse - model_rmse) / max(base_rmse, 1e-06), 
     'mae_gain_pct':100.0 * (base_mae - model_mae) / max(base_mae, 1e-06), 
     'node_gain_mean':float(np.mean(node_gain)), 
     'best_val_loss':best_val, 
     'residual_scale':args.residual_scale, 
     'cpd_transition_weight':args.cpd_transition_weight, 
     'cpd_stage_weight':args.cpd_stage_weight, 
     'rain_weight':args.rain_weight if rain_seq is not None else 0.0, 
     'rain_node_loss_weight':args.rain_node_loss_weight if rain_susceptibility is not None else 0.0, 
     'dynamic_rain_graph':int(use_dynamic_rain_graph), 
     'forecast_context':int(forecast_input is not None), 
     'event_context':int(bool(args.event_context_features)), 
     'event_context_mode':args.event_context_mode if args.event_context_features else "", 
     'event_loss_weight':args.event_loss_weight if event_sample_weight is not None else 0.0, 
     'event_loss_weight_mode':args.event_loss_weight_mode if event_sample_weight is not None else "", 
     'horizon_loss_gamma':args.horizon_loss_gamma, 
     'source_aware_residual':int(use_source_aware_residual), 
     'forecast_source_col':args.forecast_source_col if use_source_aware_residual else "", 
     'source_delta_scale':args.source_delta_scale if use_source_aware_residual else 0.0, 
     'hydro_flow_gate':int(use_dynamic_rain_graph and args.hydro_flow_gate and hydro_flow_weight is not None), 
     'temporal_encoder':args.temporal_encoder, 
     'hydro_aux_weight':args.hydro_aux_weight, 
     'hydro_attention_weight':args.hydro_attention_weight, 
     'distill_targets_dir':args.distill_targets_dir if train_teacher is not None else "", 
     'distill_weight':args.distill_weight if train_teacher is not None else 0.0, 
     'distill_scope':args.distill_scope if train_teacher is not None else "", 
     'distill_positive_only':int(args.distill_positive_only and train_teacher_weight is not None), 
     'distill_min_mean_gain':args.distill_min_mean_gain if train_teacher is not None else "", 
     'distill_sample_filter_col':args.distill_sample_filter_col if train_teacher_weight is not None else "", 
     'distill_sample_weight_col':args.distill_sample_weight_col if train_teacher_weight is not None else "", 
     'distill_sample_weight_scale':args.distill_sample_weight_scale if train_teacher_weight is not None else "", 
     'distill_horizon_start':args.distill_horizon_start if train_teacher_weight is not None else "", 
     'attention_mean':attention_mean, 
     'attention_peak_lag':attention_peak_lag, 
     'prediction_artifacts_dir':prediction_artifacts_dir, 
     'selection_metric':args.selection_metric, 
     'cpd_method':args.cpd_method}
    row.update(horizon_mae_rows(test_true_real, test_pred_real, rain_sensitive_mask, rain_top_mask))
    if split_diagnostics:
        row.update(split_diagnostics)
    row.update(event_metrics)
    return row


def write_rows(path, rows):
    os.makedirs((os.path.dirname(path)), exist_ok=True)
    fields = [
     'fold', 'train_n', 'val_n', 'test_n', 
     'train_stage', 
     'val_stage', 'test_stage', 
     'model_rmse', 'model_mae', 'persistence_rmse', 
     'persistence_mae', 
     'raw_rmse', 'raw_mae', 
     'rain_sensitive_nodes', 
     'rain_sensitive_rmse', 'rain_sensitive_mae', 
     'rain_sensitive_persistence_rmse', 
     'rain_sensitive_persistence_mae', 
     'rain_sensitive_mae_gain_pct', 
     'rain_top_nodes', 
     'rain_top_quantile', 'rain_top_rmse', 'rain_top_mae', 
     'rain_top_persistence_rmse', 
     'rain_top_persistence_mae', 
     'rain_top_mae_gain_pct', 
     'val_raw_rmse', 
     'val_raw_mae', 
     'val_cal_rmse', 'val_cal_mae', 'prediction_mode', 
     'rmse_gain_pct', 
     'mae_gain_pct', 
     'node_gain_mean', 'best_val_loss', 
     'residual_scale', 
     'cpd_transition_weight', 'cpd_stage_weight', 
     'rain_weight', 
     'rain_node_loss_weight', 'dynamic_rain_graph', 'hydro_flow_gate', 
     'forecast_context', 
     'event_context', 'event_context_mode', 'event_loss_weight', 
     'event_loss_weight_mode', 
     'horizon_loss_gamma', 
     'source_aware_residual', 'forecast_source_col', 
     'source_delta_scale', 
     'temporal_encoder', 'hydro_aux_weight', 
     'hydro_attention_weight', 
     'distill_targets_dir', 'distill_weight', 
     'distill_scope', 
     'distill_positive_only', 'distill_min_mean_gain', 
     'distill_sample_filter_col', 
     'distill_sample_weight_col', 
     'distill_sample_weight_scale', 
     'distill_horizon_start', 
     'attention_mean', 'attention_peak_lag', 
     'prediction_artifacts_dir', 
     'selection_metric', 
     'cpd_method', 
     'source_balanced_val', 
     'val_fallback_samples', 'val_official_samples', 
     'test_fallback_samples', 
     'test_official_samples']
    for h in range(1, 11):
        fields.extend([
         f"h{h}_mae",
         f"h{h}_rain_sensitive_mae",
         f"h{h}_rain_top_mae"])

    fields.extend([
     'p90_3d_samples', 'p90_3d_mae', 'p90_3d_persistence_mae', 'p90_3d_mae_gain_pct', 
     'p90_7d_samples', 
     'p90_7d_mae', 'p90_7d_persistence_mae', 'p90_7d_mae_gain_pct', 
     'p90_15d_samples', 
     'p90_15d_mae', 'p90_15d_persistence_mae', 'p90_15d_mae_gain_pct', 
     'p95_3d_samples', 
     'p95_3d_mae', 'p95_3d_persistence_mae', 'p95_3d_mae_gain_pct', 
     'p95_7d_samples', 
     'p95_7d_mae', 'p95_7d_persistence_mae', 'p95_7d_mae_gain_pct', 
     'p95_15d_samples', 
     'p95_15d_mae', 'p95_15d_persistence_mae', 'p95_15d_mae_gain_pct'])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global N_HIS
    parser = argparse.ArgumentParser(description="CPD-aware split and multi-round validation.")
    parser.add_argument("--file_path", default=(os.path.join(project_root, "dataset", "inter228_5241.csv")))
    parser.add_argument("--strategy", choices=["stratified_stage", "rolling_cpd", "both"], default="both")
    parser.add_argument("--n_his", type=int, default=12)
    parser.add_argument("--n_pred", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--node_gain_shrink", type=float, default=0.4)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--stratified_residual_scale", type=float, default=0.05)
    parser.add_argument("--rolling_residual_scale", type=float, default=0.1)
    parser.add_argument("--cpd_transition_weight", type=float, default=0.7)
    parser.add_argument("--stratified_transition_weight", type=float, default=1.2)
    parser.add_argument("--rolling_transition_weight", type=float, default=0.7)
    parser.add_argument("--cpd_stage_weight", type=float, default=0.3)
    parser.add_argument("--rainfall_features", default=None)
    parser.add_argument("--climate_features", default=None)
    parser.add_argument("--forecast_features", default=None)
    parser.add_argument("--event_context_features", default=None)
    parser.add_argument("--event_context_mode",
      choices=[
     "input", "loss_weight", "both"],
      default="input",
      help="Use event context as model input, loss-only sample weighting, or both.")
    parser.add_argument("--event_loss_weight", type=float, default=0.0)
    parser.add_argument("--event_loss_weight_mode", choices=["active", "probability"], default="active")
    parser.add_argument("--horizon_loss_gamma", type=float, default=0.0)
    parser.add_argument("--source_aware_residual", action="store_true")
    parser.add_argument("--forecast_source_col", default="forecast_is_fallback")
    parser.add_argument("--source_delta_scale", type=float, default=0.35)
    parser.add_argument("--source_balanced_val", action="store_true")
    parser.add_argument("--source_balance_col", default="forecast_is_fallback")
    parser.add_argument("--rain_susceptibility", default=None)
    parser.add_argument("--dynamic_rain_graph", action="store_true")
    parser.add_argument("--hydro_flow_gate", action="store_true")
    parser.add_argument("--temporal_encoder", choices=["none", "tcn_attention"], default="tcn_attention")
    parser.add_argument("--hydro_aux_weight", type=float, default=0.05)
    parser.add_argument("--hydro_attention_weight", type=float, default=0.02)
    parser.add_argument("--rain_weight", type=float, default=0.2)
    parser.add_argument("--rain_sensitive_quantile", type=float, default=0.6)
    parser.add_argument("--rain_node_loss_weight", type=float, default=0.0)
    parser.add_argument("--distill_targets_dir", default=None)
    parser.add_argument("--distill_weight", type=float, default=0.0)
    parser.add_argument("--distill_weight_end", type=float, default=None)
    parser.add_argument("--distill_anneal_epoch", type=int, default=15)
    parser.add_argument("--distill_scope", choices=["all", "rain_sensitive"], default="rain_sensitive")
    parser.add_argument("--distill_positive_only", action="store_true")
    parser.add_argument("--distill_min_mean_gain", type=float, default=None)
    parser.add_argument("--distill_sample_filter_col", default=None)
    parser.add_argument("--distill_sample_weight_col", default=None)
    parser.add_argument("--distill_sample_weight_scale", type=float, default=1.0)
    parser.add_argument("--distill_horizon_start", type=int, default=0)
    parser.add_argument("--distill_learnable_mask", action="store_true")
    parser.add_argument("--distill_residual_only", action="store_true", help="If true, only distill the 1x1 bypass residual delta, protecting the main STGCN graph from being distorted by the Teacher's sharp spikes.")
    parser.add_argument("--temporal_smoothness_weight", type=float, default=0.0)
    parser.add_argument("--selection_metric",
      choices=[
     "loss", "rain_sensitive_mae", "val_real_mae", "val_real_rain_sensitive_mae"],
      default="loss")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpd_method", choices=(list(CPD_METHODS)), default="binseg")
    parser.add_argument("--cpd_mode", choices=["mean", "std", "top", "mix"], default="mix")
    parser.add_argument("--cpd_cost", default="l2")
    parser.add_argument("--cpd_penalty", type=float, default=10)
    parser.add_argument("--cpd_top_ratio", type=float, default=0.1)
    parser.add_argument("--cpd_min_size", type=int, default=10)
    parser.add_argument("--pelt_penalty", type=float, default=None)
    parser.add_argument("--bfast_frequency", type=int, default=23)
    parser.add_argument("--use_fold_recommendations", action="store_true", default=True)
    parser.add_argument("--fold_filter", default=None, help="Comma-separated fold names such as stratified_stage,cp_180.")
    parser.add_argument("--max_rounds", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--save_fold_predictions", action="store_true")
    parser.add_argument("--output_dir", default=(os.path.join(project_root, "output", "cpd_split_validation")))
    args = parser.parse_args()
    np.random.seed(args.seed)
    N_HIS = args.n_his
    raw_seq, coords, elevation, time_cols, node_ids = load_and_clean_data(args.file_path)
    stage_seq, change_points = detect_change_points_professional(raw_seq,
      penalty=(args.cpd_penalty),
      mode=(args.cpd_mode),
      top_ratio=(args.cpd_top_ratio),
      min_size=(args.cpd_min_size),
      method=(args.cpd_method),
      model=(args.cpd_cost),
      pelt_penalty=(args.pelt_penalty),
      bfast_frequency=(args.bfast_frequency))
    rain_seq = None
    aligned_rain = None
    rain_cols = None
    if args.rainfall_features:
        aligned_rain = align_rainfall_to_insar((args.rainfall_features),
          time_cols,
          climate_csv=(args.climate_features))
        rain_seq, rain_cols = normalized_rain_matrix(aligned_rain)
        print(f">> Loaded rainfall features for CPD split validation: {list(rain_cols)}")
        if args.climate_features:
            print(f">> Merged daily climate/hydro features: {args.climate_features}")
    rain_susceptibility = load_rain_susceptibility(args.rain_susceptibility, node_ids, raw_seq.shape[1])
    if args.dynamic_rain_graph:
        if rain_seq is None or rain_susceptibility is None:
            print(">> dynamic_rain_graph requested but rainfall features or rain_susceptibility is missing; branch will stay disabled.")
    starts = np.arange(0, (raw_seq.shape[0] - args.n_his - args.n_pred + 1), dtype=(np.int32))
    forecast_context = None
    forecast_feature_cols = None
    if args.forecast_features:
        forecast_df = forecast_context_for_starts(args.forecast_features, time_cols, starts, args.n_his)
        forecast_feature_cols = [c for c in forecast_df.columns if c != "origin_date"]
        forecast_context = forecast_df[forecast_feature_cols].to_numpy(dtype=(np.float32))
        print(f">> Loaded forecast context features: {forecast_feature_cols}")
    event_context = None
    event_feature_cols = None
    if args.event_context_features:
        event_df = event_context_for_starts(args.event_context_features, time_cols, starts, args.n_his)
        event_feature_cols = list(event_df.columns)
        event_context = event_df[event_feature_cols].to_numpy(dtype=(np.float32))
        print(f">> Loaded event context features: {event_feature_cols}")
    if event_context is not None:
        if args.event_context_mode in ('input', 'both'):
            if forecast_context is None:
                forecast_context = event_context
                forecast_feature_cols = event_feature_cols
        else:
            forecast_context = np.concatenate([forecast_context, event_context], axis=1)
            forecast_feature_cols = list(forecast_feature_cols or []) + list(event_feature_cols)
    target_stage = target_stage_for_windows(stage_seq, starts, args.n_his, args.n_pred)
    folds = []
    if args.strategy in ('stratified_stage', 'both'):
        train, val, test = split_stratified_stage(starts, target_stage)
        folds.append(("stratified_stage", train, val, test))
    if args.strategy in ('rolling_cpd', 'both'):
        folds.extend(split_rolling_cpd(starts, change_points, args.n_his, args.n_pred))
    if args.fold_filter:
        requested_folds = {name.strip() for name in args.fold_filter.split(",") if name.strip()}
        folds = [fold for fold in folds if fold[0] in requested_folds]
    if args.max_rounds > 0:
        folds = folds[:args.max_rounds]
    split_diagnostics = {}
    if args.source_balanced_val:
        folds, split_diagnostics = rebalance_validation_by_source(folds, forecast_context, forecast_feature_cols, args.source_balance_col, args.n_his, args.n_pred)
    print("\nCPD-aware split plan:")
    for name, train, val, test in folds:
        source_msg = ""
        if name in split_diagnostics:
            diag = split_diagnostics[name]
            source_msg = f', val_source=fallback:{diag["val_fallback_samples"]}/official:{diag["val_official_samples"]}, test_source=fallback:{diag["test_fallback_samples"]}/official:{diag["test_official_samples"]}'
        print(f"- {name}: train={len(train)} [{stage_histogram(stage_seq, train, args.n_his, args.n_pred)}], val={len(val)} [{stage_histogram(stage_seq, val, args.n_his, args.n_pred)}], test={len(test)} [{stage_histogram(stage_seq, test, args.n_his, args.n_pred)}]{source_msg}")

    if args.dry_run:
        return
    adj = build_adjacency_matrix(coords, elevation, threshold=0.23, knn_k=12)
    hydro_flow_weight = build_hydro_flow_weight(adj, elevation)
    rows = []
    for name, train, val, test in folds:
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            print(f"Skipping {name}: empty split.")
        else:
            fold_args = copy(args)
            if args.use_fold_recommendations:
                if name == "stratified_stage":
                    fold_args.residual_scale = args.stratified_residual_scale
                    fold_args.cpd_transition_weight = args.stratified_transition_weight
                else:
                    fold_args.residual_scale = args.rolling_residual_scale
                    fold_args.cpd_transition_weight = args.rolling_transition_weight
                print(f">> Fold params for {name}: residual_scale={fold_args.residual_scale}, cpd_transition_weight={fold_args.cpd_transition_weight}, cpd_stage_weight={fold_args.cpd_stage_weight}")
                row = train_one_fold(name,
                  raw_seq,
                  stage_seq,
                  adj,
                  train,
                  val,
                  test,
                  fold_args,
                  rain_seq=rain_seq,
                  rain_feature_cols=(rain_cols if rain_seq is not None else None),
                  rain_susceptibility=rain_susceptibility,
                  hydro_flow_weight=hydro_flow_weight,
                  aligned_rain=aligned_rain,
                  forecast_context=forecast_context,
                  forecast_feature_cols=forecast_feature_cols,
                  event_context=event_context,
                  event_feature_cols=event_feature_cols,
                  split_diagnostics=(split_diagnostics.get(name)))
                if row is not None:
                    rows.append(row)
                    write_rows(os.path.join(args.output_dir, "cpd_split_results.csv"), rows)

    result_path = os.path.join(args.output_dir, "cpd_split_results.csv")
    write_rows(result_path, rows)
    print(f"\nSaved CPD split validation report: {result_path}")


if __name__ == "__main__":
    main()

