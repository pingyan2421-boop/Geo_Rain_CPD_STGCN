import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

CPD_METHODS = (
    "binseg",
    "pelt",
    "bfast",
    "kernelcpd",
    "dynp",
    "bottomup",
    "window",
    "bocpd",
    "mdl_multivariate",
)


def build_cpd_signal(data_seq, mode="mix", top_ratio=0.10):
    """Build the normalized temporal signal used by broad-stage CPD."""
    t_len, n_nodes = data_seq.shape
    top_n = max(1, int(n_nodes * top_ratio))
    node_motion = np.std(np.diff(data_seq, axis=0), axis=0)
    active_idx = np.argsort(node_motion)[-top_n:]

    mean_signal = np.mean(data_seq, axis=1)
    std_signal = np.std(data_seq, axis=1)
    top_signal = np.mean(data_seq[:, active_idx], axis=1)

    signals = {
        "mean": mean_signal,
        "std": std_signal,
        "top": top_signal,
        "mix": np.stack([mean_signal, std_signal, top_signal], axis=1),
    }
    signal = signals.get(mode, signals["mix"])
    if signal.ndim == 1:
        signal = signal[:, None]
    return (signal - signal.mean(axis=0, keepdims=True)) / (signal.std(axis=0, keepdims=True) + 1e-6)


def stages_from_change_points(t_len, change_points):
    cps = sorted(set(int(cp) for cp in change_points if 0 < int(cp) < t_len))
    stages = np.zeros(t_len, dtype=np.int32)
    start = 0
    for stage_id, end in enumerate(cps + [t_len]):
        stages[start:end] = stage_id
        start = end
    return stages, cps


def _filter_change_points(cps, t_len, min_size):
    return sorted(set(int(cp) for cp in cps if min_size <= int(cp) <= t_len - min_size))


def _detect_binseg(signal, penalty, min_size, model):
    import ruptures as rpt

    max_bkps = _max_bkps_from_penalty(penalty)
    bkps = rpt.Binseg(model=model, min_size=min_size).fit(signal).predict(n_bkps=max_bkps)
    return bkps[:-1]


def _max_bkps_from_penalty(penalty):
    return 6 if penalty <= 5 else 5 if penalty <= 10 else 4 if penalty <= 15 else 3


def _predict_fixed_bkps(algo, penalty):
    return algo.predict(n_bkps=_max_bkps_from_penalty(penalty))[:-1]


def _detect_kernelcpd(signal, penalty, min_size):
    import ruptures as rpt

    return _predict_fixed_bkps(rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(signal), penalty)


def _detect_dynp(signal, penalty, min_size, model):
    import ruptures as rpt

    return _predict_fixed_bkps(rpt.Dynp(model=model, min_size=min_size, jump=1).fit(signal), penalty)


def _detect_bottomup(signal, penalty, min_size, model):
    import ruptures as rpt

    return _predict_fixed_bkps(rpt.BottomUp(model=model, min_size=min_size, jump=1).fit(signal), penalty)


def _detect_window(signal, penalty, min_size, model):
    import ruptures as rpt

    width = max(2 * int(min_size), min(40, signal.shape[0] // 4))
    return _predict_fixed_bkps(rpt.Window(width=width, model=model, min_size=min_size, jump=1).fit(signal), penalty)


def _detect_pelt(signal, penalty, min_size, model, pelt_penalty):
    import ruptures as rpt

    pen = float(pelt_penalty if pelt_penalty is not None else penalty)
    bkps = rpt.Pelt(model=model, min_size=min_size).fit(signal).predict(pen=pen)
    return bkps[:-1]


def _find_rscript(rscript_path=None):
    candidates = [
        rscript_path,
        shutil.which("Rscript"),
        r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe",
        r"C:\Program Files\R\R-4.6.0\bin\x64\Rscript.exe",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise RuntimeError("Rscript not found. Install R and the R package 'bfast' before using --cpd_method bfast.")


def _signal_to_univariate(signal):
    if signal.ndim == 1 or signal.shape[1] == 1:
        return np.asarray(signal).reshape(-1)
    return np.mean(signal, axis=1)


def _detect_bfast(signal, min_size, frequency, rscript_path=None, h=0.15, max_iter=5):
    y = _signal_to_univariate(signal)
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_csv = os.path.join(tmp_dir, "signal.csv")
        output_csv = os.path.join(tmp_dir, "breakpoints.csv")
        script_path = os.path.join(tmp_dir, "run_bfast.R")
        pd.DataFrame({"signal": y}).to_csv(input_csv, index=False)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(
                "args <- commandArgs(trailingOnly=TRUE)\n"
                "input <- args[1]\n"
                "output <- args[2]\n"
                "freq <- as.integer(args[3])\n"
                "hval <- as.numeric(args[4])\n"
                "max_iter <- as.integer(args[5])\n"
                "suppressPackageStartupMessages(library(bfast))\n"
                "df <- read.csv(input)\n"
                "y <- as.numeric(df$signal)\n"
                "y <- y[is.finite(y)]\n"
                "if (length(y) < 20) {\n"
                "  write.csv(data.frame(change_point_index=integer()), output, row.names=FALSE)\n"
                "  quit(save='no')\n"
                "}\n"
                "fit <- bfast(ts(y, frequency=freq), season='none', h=hval, max.iter=max_iter)\n"
                "latest <- fit$output[[length(fit$output)]]\n"
                "cps <- latest$Vt.bp\n"
                "cps <- cps[is.finite(cps)]\n"
                "cps <- as.integer(round(cps))\n"
                "write.csv(data.frame(change_point_index=cps), output, row.names=FALSE)\n"
            )
        env = os.environ.copy()
        env.setdefault("R_LIBS_USER", os.path.join(os.path.expanduser("~"), "R", "win-library", "4.6"))
        cmd = [_find_rscript(rscript_path), script_path, input_csv, output_csv, str(int(frequency)), str(float(h)), str(int(max_iter))]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"BFAST failed: {proc.stderr.strip() or proc.stdout.strip()}")
        if not os.path.exists(output_csv):
            return []
        out = pd.read_csv(output_csv)
        if "change_point_index" not in out.columns:
            return []
        return out["change_point_index"].dropna().astype(int).tolist()


def _segment_sse(signal, start, end):
    segment = signal[start:end]
    if len(segment) == 0:
        return 0.0
    centered = segment - np.mean(segment, axis=0, keepdims=True)
    return float(np.sum(centered * centered))


def _detect_mdl_multivariate(signal, penalty, min_size):
    t_len, n_dim = signal.shape
    scale = max(0.5, float(penalty) / 10.0)
    penalty_value = scale * n_dim * np.log(max(t_len, 2))
    segments = [(0, t_len)]
    cps = []
    max_bkps = _max_bkps_from_penalty(penalty)

    while len(cps) < max_bkps:
        best = None
        for seg_idx, (start, end) in enumerate(segments):
            if end - start < 2 * min_size:
                continue
            base_cost = _segment_sse(signal, start, end)
            for cp in range(start + min_size, end - min_size + 1):
                split_cost = _segment_sse(signal, start, cp) + _segment_sse(signal, cp, end)
                gain = base_cost - split_cost - penalty_value
                if best is None or gain > best[0]:
                    best = (gain, seg_idx, cp)
        if best is None or best[0] <= 0.0:
            break
        _, seg_idx, cp = best
        start, end = segments.pop(seg_idx)
        segments.extend([(start, cp), (cp, end)])
        segments.sort()
        cps.append(cp)
    return sorted(cps)


def _detect_bocpd_proxy(signal, penalty, min_size):
    y = _signal_to_univariate(signal)
    t_len = len(y)
    window = max(int(min_size), min(20, t_len // 6))
    scores = np.zeros(t_len, dtype=np.float64)
    for cp in range(window, t_len - window):
        left = y[cp - window:cp]
        right = y[cp:cp + window]
        pooled = np.var(np.concatenate([left, right])) + 1e-6
        scores[cp] = abs(np.mean(right) - np.mean(left)) / np.sqrt(pooled)

    max_bkps = _max_bkps_from_penalty(penalty)
    threshold = np.quantile(scores[scores > 0], 0.75) if np.any(scores > 0) else np.inf
    candidates = np.argsort(scores)[::-1]
    cps = []
    for cp in candidates:
        if scores[cp] < threshold:
            break
        if cp < min_size or cp > t_len - min_size:
            continue
        if all(abs(int(cp) - existing) >= min_size for existing in cps):
            cps.append(int(cp))
        if len(cps) >= max_bkps:
            break
    return sorted(cps)


def _detect_fast_bocpd_external(signal, penalty, min_size):
    y = _signal_to_univariate(signal)
    t_len = len(y)
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_csv = os.path.join(tmp_dir, "signal.csv")
        output_csv = os.path.join(tmp_dir, "bocpd.csv")
        script_path = os.path.join(tmp_dir, "run_fast_bocpd.py")
        pd.DataFrame({"signal": y}).to_csv(input_csv, index=False)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(
                "import sys\n"
                "import numpy as np\n"
                "import pandas as pd\n"
                "from fast_bocpd import BOCPD, ConstantHazard, GaussianNIG\n"
                "input_csv, output_csv, min_size, max_bkps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])\n"
                "y = pd.read_csv(input_csv)['signal'].to_numpy(dtype=float)\n"
                "y = y[np.isfinite(y)]\n"
                "if len(y) < max(20, 2 * min_size):\n"
                "    pd.DataFrame({'change_point_index': []}).to_csv(output_csv, index=False)\n"
                "    raise SystemExit(0)\n"
                "model = BOCPD(GaussianNIG(mu0=float(np.mean(y)), kappa0=1.0, alpha0=1.0, beta0=float(np.var(y) + 1e-6)), ConstantHazard(lambda_=max(20.0, len(y) / max(1, max_bkps))), max_run_length=len(y) + 5)\n"
                "cp_probs = model.batch_update(y)\n"
                "threshold = np.quantile(cp_probs[min_size:-min_size], 0.90) if len(y) > 2 * min_size else np.inf\n"
                "order = np.argsort(cp_probs)[::-1]\n"
                "cps = []\n"
                "for cp in order:\n"
                "    cp = int(cp)\n"
                "    if cp < min_size or cp > len(y) - min_size or cp_probs[cp] < threshold:\n"
                "        continue\n"
                "    if all(abs(cp - old) >= min_size for old in cps):\n"
                "        cps.append(cp)\n"
                "    if len(cps) >= max_bkps:\n"
                "        break\n"
                "pd.DataFrame({'change_point_index': sorted(cps)}).to_csv(output_csv, index=False)\n"
            )
        cmd = ["py", "-3.11", script_path, input_csv, output_csv, str(int(min_size)), str(_max_bkps_from_penalty(penalty))]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        out = pd.read_csv(output_csv)
        return out["change_point_index"].dropna().astype(int).tolist() if "change_point_index" in out.columns else []


def detect_change_points_by_method(
    data_seq,
    penalty=10,
    mode="mix",
    top_ratio=0.10,
    min_size=10,
    method="binseg",
    model="l2",
    pelt_penalty=None,
    bfast_frequency=23,
    rscript_path=None,
):
    signal = build_cpd_signal(data_seq, mode=mode, top_ratio=top_ratio)
    t_len = data_seq.shape[0]
    method = (method or "binseg").lower()
    if method == "binseg":
        cps = _detect_binseg(signal, penalty=penalty, min_size=min_size, model=model)
    elif method == "pelt":
        cps = _detect_pelt(signal, penalty=penalty, min_size=min_size, model=model, pelt_penalty=pelt_penalty)
    elif method == "bfast":
        cps = _detect_bfast(signal, min_size=min_size, frequency=bfast_frequency, rscript_path=rscript_path)
    elif method == "kernelcpd":
        cps = _detect_kernelcpd(signal, penalty=penalty, min_size=min_size)
    elif method == "dynp":
        cps = _detect_dynp(signal, penalty=penalty, min_size=min_size, model=model)
    elif method == "bottomup":
        cps = _detect_bottomup(signal, penalty=penalty, min_size=min_size, model=model)
    elif method == "window":
        cps = _detect_window(signal, penalty=penalty, min_size=min_size, model=model)
    elif method == "bocpd":
        try:
            cps = _detect_fast_bocpd_external(signal, penalty=penalty, min_size=min_size)
        except Exception as exc:
            print(f">> fast_bocpd external call failed ({exc}); using BOCPD proxy.")
            cps = _detect_bocpd_proxy(signal, penalty=penalty, min_size=min_size)
    elif method == "mdl_multivariate":
        cps = _detect_mdl_multivariate(signal, penalty=penalty, min_size=min_size)
    else:
        raise ValueError(f"Unsupported CPD method: {method}")
    return _filter_change_points(cps, t_len, min_size)
