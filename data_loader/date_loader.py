import os
import numpy as np
import pandas as pd

from data_loader.rainfall_loader import align_rainfall_to_insar, normalized_rain_matrix
from data_loader.cpd_methods import detect_change_points_by_method, stages_from_change_points


class Dataset(object):
    def __init__(self, data, stats, adj, stages, meta=None):
        self.data = data
        self.stats = stats
        self.adj = adj
        self.stages = stages
        self.meta = meta or {}

    def get_data(self, name):
        return self.data[name]

    def get_stage(self, name):
        return self.stages[name]

    def get_len(self, name):
        return len(self.data[name])

    def get_stats(self):
        return self.stats

    def get_adj(self):
        return self.adj

    def get_meta(self):
        return self.meta

    def get_rain(self, name):
        rainfall = self.meta.get("rainfall")
        if rainfall is None:
            return None
        return rainfall.get(name)

    def get_rain_susceptibility(self):
        return self.meta.get("rain_susceptibility")

    def get_hydro_flow_weight(self):
        return self.meta.get("hydro_flow_weight")


def load_and_clean_data(file_path):
    """Load the paper-style landslide displacement table.

    Expected shape:
      rows: spatial measurement nodes
      columns: node id, longitude, latitude, elevation, time observations...

    Returns
      data_seq: [T, N] displacement sequence in physical units
      coords: [N, 2] longitude/latitude
      elevation: [N]
      time_cols: list[str]
      node_ids: list[str]
    """
    df = pd.read_csv(file_path)
    required = ["date", "i", "j", "h"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    time_cols = list(df.columns[4:])
    if len(time_cols) < 20:
        raise ValueError("Too few temporal columns. Use dataset/inter228_5241.csv, not the transposed *_5241a.csv file.")

    values = df[time_cols].apply(pd.to_numeric, errors="coerce")
    values = values.interpolate(axis=1, limit_direction="both")
    values = values.fillna(values.mean(axis=1))
    values = values.fillna(0.0)

    data_seq = values.to_numpy(dtype=np.float32).T
    coords = df[["i", "j"]].to_numpy(dtype=np.float32)
    elevation = pd.to_numeric(df["h"], errors="coerce").fillna(df["h"].median()).to_numpy(dtype=np.float32)
    node_ids = df["date"].astype(str).tolist()

    print(f">> Loaded paper-format data: T={data_seq.shape[0]}, N={data_seq.shape[1]}")
    print(f">> Physical displacement range: {data_seq.min():.3f} to {data_seq.max():.3f} mm")
    return data_seq, coords, elevation, time_cols, node_ids


def build_adjacency_matrix(coords, elevation, threshold=0.23, knn_k=12):
    """Build a row-normalized terrain-aware KNN graph.

    The model still consumes a dense TensorFlow graph kernel, but only KNN edges
    are non-zero. This keeps the physical graph local and avoids the old fully
    connected spatial smoothing.
    """
    n_nodes = coords.shape[0]
    k = int(max(2, min(knn_k + 1, n_nodes)))

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)
    except Exception:
        from scipy.spatial import cKDTree
        distances, indices = cKDTree(coords).query(coords, k=k)

    neighbor_dist = distances[:, 1:].reshape(-1)
    neighbor_dist = neighbor_dist[neighbor_dist > 0]
    sigma_d = np.median(neighbor_dist) if neighbor_dist.size else 1.0
    sigma_d = max(float(sigma_d), 1e-6)

    elev_diff_samples = np.abs(elevation[:, None] - elevation[indices[:, 1:]]).reshape(-1)
    sigma_h = np.median(elev_diff_samples[elev_diff_samples > 0]) if np.any(elev_diff_samples > 0) else 1.0
    sigma_h = max(float(sigma_h), 1e-6)

    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for src in range(n_nodes):
        for dist, dst in zip(distances[src, 1:], indices[src, 1:]):
            dh = float(elevation[src] - elevation[dst])
            distance_weight = np.exp(-(float(dist) ** 2) / (2.0 * sigma_d ** 2))
            elevation_weight = np.exp(-(dh ** 2) / (2.0 * sigma_h ** 2))
            downslope_weight = 1.0 if dh >= 0.0 else 0.35
            weight = distance_weight * elevation_weight * downslope_weight
            if weight >= threshold:
                adj[src, dst] = max(adj[src, dst], weight)

    # Keep every node connected even under a strict threshold.
    fallback_edges = 0
    for src in range(n_nodes):
        if np.count_nonzero(adj[src]) == 0:
            dst = indices[src, 1]
            adj[src, dst] = 0.25
            fallback_edges += 1

    np.fill_diagonal(adj, 1.0)
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    adj = adj / row_sum

    density = np.count_nonzero(adj) / float(n_nodes * n_nodes)
    print(f">> Built KNN terrain graph: N={n_nodes}, k={knn_k}, density={density:.5f}, fallback_rows={fallback_edges}")
    return adj.astype(np.float32)


def build_hydro_flow_weight(adj, elevation):
    """Estimate a node-level downslope flow weight from the terrain graph."""
    adj = np.asarray(adj, dtype=np.float32)
    elevation = np.asarray(elevation, dtype=np.float32)
    if adj.shape[0] != elevation.shape[0]:
        raise ValueError("Adjacency and elevation length mismatch when building hydro flow weights.")

    elev_drop = elevation[:, None] - elevation[None, :]
    downslope = np.where(elev_drop > 0.0, elev_drop, 0.0).astype(np.float32)
    raw = np.sum(adj * downslope, axis=1)
    if np.nanmax(raw) > np.nanmin(raw):
        raw = (raw - np.nanmin(raw)) / (np.nanmax(raw) - np.nanmin(raw))
    else:
        raw = np.zeros_like(raw, dtype=np.float32)
    weight = np.clip(0.5 + 0.5 * raw, 0.5, 1.0).astype(np.float32)
    print(f">> Built hydro flow weights: range=({weight.min():.3f}, {weight.max():.3f})")
    return weight


def detect_change_points_professional(data_seq, penalty=10, mode="mix", top_ratio=0.10, min_size=10,
                                      method="binseg", model="l2", pelt_penalty=None,
                                      bfast_frequency=23, rscript_path=None):
    """Detect broad temporal deformation stages from the correctly oriented series."""
    t_len = data_seq.shape[0]
    try:
        cps = detect_change_points_by_method(
            data_seq,
            penalty=penalty,
            mode=mode,
            top_ratio=top_ratio,
            min_size=min_size,
            method=method,
            model=model,
            pelt_penalty=pelt_penalty,
            bfast_frequency=bfast_frequency,
            rscript_path=rscript_path,
        )
    except Exception as exc:
        print(f">> CPD method {method} unavailable or failed ({exc}); using quantile stage splits.")
        cps = [int(t_len * q) for q in (0.25, 0.5, 0.75)]

    stages, cps = stages_from_change_points(t_len, cps)
    print(f">> CPD method: {method}")
    print(f">> Change points: {cps}")
    print(f">> Stage distribution: {dict(zip(*np.unique(stages, return_counts=True)))}")
    return stages, cps


def seq_gen(length, data_seq, stage_seq, start_idx, n_his, n_frame):
    """Generate sliding windows from [T, N] into [samples, n_frame, N, 1]."""
    if length < n_frame:
        raise ValueError(f"Segment length {length} is shorter than n_frame {n_frame}.")

    samples = length - n_frame + 1
    x = np.zeros((samples, n_frame, data_seq.shape[1], 1), dtype=np.float32)
    s = np.zeros((samples, n_his), dtype=np.int32)
    for i in range(samples):
        st = start_idx + i
        ed = st + n_frame
        x[i, :, :, 0] = data_seq[st:ed]
        s[i] = stage_seq[st:st + n_his]
    return x, s


def rain_seq_gen(length, rain_seq, start_idx, n_his, n_frame):
    if rain_seq is None:
        return None
    if length < n_frame:
        raise ValueError(f"Segment length {length} is shorter than n_frame {n_frame}.")

    samples = length - n_frame + 1
    x = np.zeros((samples, n_his, rain_seq.shape[1]), dtype=np.float32)
    for i in range(samples):
        st = start_idx + i
        x[i] = rain_seq[st:st + n_his]
    return x


def load_rain_susceptibility(path, node_ids, n_nodes):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Rain susceptibility file not found: {path}")

    df = pd.read_csv(path)
    if "rain_susceptibility" not in df.columns:
        raise ValueError("Rain susceptibility CSV must contain a rain_susceptibility column.")

    values = np.zeros(int(n_nodes), dtype=np.float32)
    filled = np.zeros(int(n_nodes), dtype=bool)
    if "node_index" in df.columns:
        for row in df.itertuples(index=False):
            idx = int(getattr(row, "node_index"))
            if 0 <= idx < n_nodes:
                values[idx] = float(getattr(row, "rain_susceptibility"))
                filled[idx] = True
    elif "node_id" in df.columns:
        node_to_idx = {str(node_id): i for i, node_id in enumerate(node_ids)}
        for row in df.itertuples(index=False):
            idx = node_to_idx.get(str(getattr(row, "node_id")))
            if idx is not None:
                values[idx] = float(getattr(row, "rain_susceptibility"))
                filled[idx] = True
    elif len(df) == n_nodes:
        values[:] = pd.to_numeric(df["rain_susceptibility"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        filled[:] = True
    else:
        raise ValueError("Rain susceptibility CSV needs node_index, node_id, or exactly one row per node.")

    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0).astype(np.float32)
    print(f">> Loaded rain susceptibility: {int(filled.sum())}/{n_nodes} nodes from {path}")
    return values


def data_gen(file_path, data_config, save_path, n_his, n_pred, cpd_penalty=10,
             cpd_mode="mix", cpd_top_ratio=0.10, cpd_min_size=10, adj_threshold=0.23,
             rainfall_features_path=None, rain_susceptibility_path=None, cpd_method="binseg",
             cpd_cost="l2", pelt_penalty=None, bfast_frequency=23):
    raw_seq, coords, elevation, time_cols, node_ids = load_and_clean_data(file_path)
    n_train, n_val, n_test = map(int, data_config)
    total = n_train + n_val + n_test
    if total != raw_seq.shape[0]:
        raise ValueError(f"Train/val/test split {data_config} sums to {total}, but data has T={raw_seq.shape[0]}.")

    n_frame = int(n_his) + int(n_pred)
    if min(data_config) < n_frame:
        raise ValueError(f"Each split must be at least n_his+n_pred={n_frame}. Got {data_config}.")

    train_raw = raw_seq[:n_train]
    mean = np.mean(train_raw, dtype=np.float64).astype(np.float32)
    std = np.std(train_raw, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    norm_seq = ((raw_seq - mean) / std).astype(np.float32)

    stage_seq, change_points = detect_change_points_professional(
        raw_seq,
        penalty=cpd_penalty,
        mode=cpd_mode,
        top_ratio=cpd_top_ratio,
        min_size=cpd_min_size,
        method=cpd_method,
        model=cpd_cost,
        pelt_penalty=pelt_penalty,
        bfast_frequency=bfast_frequency,
    )
    adj = build_adjacency_matrix(coords, elevation, threshold=adj_threshold, knn_k=12)
    hydro_flow_weight = build_hydro_flow_weight(adj, elevation)

    train_x, train_s = seq_gen(n_train, norm_seq, stage_seq, 0, n_his, n_frame)
    val_x, val_s = seq_gen(n_val, norm_seq, stage_seq, n_train, n_his, n_frame)
    test_x, test_s = seq_gen(n_test, norm_seq, stage_seq, n_train + n_val, n_his, n_frame)

    rain_seq = None
    rain_feature_cols = np.array([], dtype=object)
    aligned_rain = None
    if rainfall_features_path:
        aligned_rain = align_rainfall_to_insar(rainfall_features_path, time_cols)
        rain_seq, rain_feature_cols = normalized_rain_matrix(aligned_rain)
        print(f">> Loaded rainfall features: {rainfall_features_path}")
        print(f">> Rainfall feature columns: {list(rain_feature_cols)}")

    train_rain = rain_seq_gen(n_train, rain_seq, 0, n_his, n_frame)
    val_rain = rain_seq_gen(n_val, rain_seq, n_train, n_his, n_frame)
    test_rain = rain_seq_gen(n_test, rain_seq, n_train + n_val, n_his, n_frame)
    rain_susceptibility = load_rain_susceptibility(rain_susceptibility_path, node_ids, raw_seq.shape[1])

    data = {"train": train_x, "val": val_x, "test": test_x}
    stages = {"train": train_s, "val": val_s, "test": test_s}
    stats = {"mean": mean, "std": std}
    rainfall = None
    if rain_seq is not None:
        rainfall = {"train": train_rain, "val": val_rain, "test": test_rain}
    meta = {
        "source_file": os.path.abspath(file_path),
        "orientation": "time_by_node",
        "n_time": int(raw_seq.shape[0]),
        "n_nodes": int(raw_seq.shape[1]),
        "time_cols": np.array(time_cols, dtype=object),
        "node_ids": np.array(node_ids, dtype=object),
        "change_points": np.array(change_points, dtype=np.int32),
        "cpd_method": cpd_method,
        "coords": coords.astype(np.float32),
        "elevation": elevation.astype(np.float32),
        "rainfall": rainfall,
        "rain_feature_cols": rain_feature_cols,
        "rain_susceptibility": rain_susceptibility,
        "hydro_flow_weight": hydro_flow_weight,
    }

    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, "preprocessed_data.npz")
    np.savez_compressed(
        out_file,
        train=train_x,
        val=val_x,
        test=test_x,
        train_stage=train_s,
        val_stage=val_s,
        test_stage=test_s,
        adj=adj,
        mean=mean,
        std=std,
        source_file=meta["source_file"],
        orientation=meta["orientation"],
        n_time=meta["n_time"],
        n_nodes=meta["n_nodes"],
        time_cols=meta["time_cols"],
        node_ids=meta["node_ids"],
        change_points=meta["change_points"],
        cpd_method=np.array(meta["cpd_method"], dtype=object),
        coords=meta["coords"],
        elevation=meta["elevation"],
        rain_feature_cols=meta["rain_feature_cols"],
        rain_susceptibility=np.array([], dtype=np.float32) if rain_susceptibility is None else rain_susceptibility,
        hydro_flow_weight=meta["hydro_flow_weight"],
    )
    if rain_seq is not None:
        with np.load(out_file, allow_pickle=True) as pack:
            payload = {key: pack[key] for key in pack.files}
        payload.update({
            "train_rain": train_rain,
            "val_rain": val_rain,
            "test_rain": test_rain,
            "aligned_rainfall": aligned_rain.to_numpy(dtype=object),
            "aligned_rainfall_columns": aligned_rain.columns.to_numpy(dtype=object),
        })
        np.savez_compressed(out_file, **payload)
    print(f">> Saved preprocessed data to {out_file}")
    print(f">> train/val/test windows: {train_x.shape}, {val_x.shape}, {test_x.shape}")
    return Dataset(data, stats, adj, stages, meta)


def load_preprocessed(save_path):
    file_path = os.path.join(save_path, "preprocessed_data.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessed file not found: {file_path}")

    pack = np.load(file_path, allow_pickle=True)
    data = {
        "train": pack["train"].astype(np.float32),
        "val": pack["val"].astype(np.float32),
        "test": pack["test"].astype(np.float32),
    }
    stages = {
        "train": pack["train_stage"].astype(np.int32),
        "val": pack["val_stage"].astype(np.int32),
        "test": pack["test_stage"].astype(np.int32),
    }
    stats = {
        "mean": np.float32(pack["mean"]),
        "std": np.float32(pack["std"]),
    }
    meta = {k: pack[k] for k in pack.files if k not in {
        "train", "val", "test", "train_stage", "val_stage", "test_stage", "adj", "mean", "std",
        "train_rain", "val_rain", "test_rain"
    }}
    if {"train_rain", "val_rain", "test_rain"}.issubset(set(pack.files)):
        meta["rainfall"] = {
            "train": pack["train_rain"].astype(np.float32),
            "val": pack["val_rain"].astype(np.float32),
            "test": pack["test_rain"].astype(np.float32),
        }
    else:
        meta["rainfall"] = None
    if "rain_susceptibility" in pack.files and pack["rain_susceptibility"].size:
        meta["rain_susceptibility"] = pack["rain_susceptibility"].astype(np.float32)
    else:
        meta["rain_susceptibility"] = None
    if "hydro_flow_weight" in pack.files and pack["hydro_flow_weight"].size:
        meta["hydro_flow_weight"] = pack["hydro_flow_weight"].astype(np.float32)
    else:
        meta["hydro_flow_weight"] = None
    return Dataset(data, stats, pack["adj"].astype(np.float32), stages, meta)
