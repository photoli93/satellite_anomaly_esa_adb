"""
Purpose : Simulate a live ground-station feed

Run in local, preparing a small, realistic telemetry segment
for deployment/telemetry_replay.py to stream on the Jetson.

X_test_starter.npy holds real ESA telemetry (file comes from raw file from train.parquet,
split by the Kaggle competition itself before any of the preprocessing). It's pre-scaled,
sliding windows with stride=1 so consecutive windows overlap by all but one timestep and
the underlying raw timestep sequence can be reconstructed exactly by reading each window's
first timestep, plus the final window's remaining tail.

Rather than ship pre-scaled overlapping windows, this script reconstructs the original
(pre-StandardScaler) telemetry values via scaler.inverse_transform, finds the longest run
of windows the model itself flags as anomalous and exports a contiguous raw segment
around that run so telemetry_replay.py can rescale + slide a window buffer live on the
Jetson, mirroring exactly how a real onboard system would receive raw sensor ticks.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort


def score_all_windows(session, input_name, X, batch_size):
    n = X.shape[0]
    probs = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.ascontiguousarray(X[start:end], dtype=np.float32)
        logits = session.run(None, {input_name: batch})[0]
        probs[start:end] = 1 / (1 + np.exp(-logits))
    return probs


def longest_run(flags):
    best_start, best_len = 0, 0
    run_start, run_len = 0, 0
    for i, flag in enumerate(flags):
        if flag:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    return best_start, best_len


def main():
    parser = argparse.ArgumentParser(description="Prepare a realistic telemetry replay segment from real test data")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/demo"))
    parser.add_argument("--context-before", type=int, default=150, help="Raw timesteps of normal context before the flagged run")
    parser.add_argument("--context-after", type=int, default=100, help="Raw timesteps of normal context after the flagged run")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    with open(args.bundle_dir / "deployment_config.json") as f:
        config = json.load(f)

    scaler = joblib.load(args.bundle_dir / config["scaler_file"])
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # suppress the harmless static-output-shape warning on every batch
    session = ort.InferenceSession(str(args.bundle_dir / config["model_file"]), sess_options=session_options,
                                    providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    window_size = config["window_size"]
    threshold = config["best_threshold"]

    X_test = np.load(args.data_dir / "X_test_starter.npy", mmap_mode="r")
    n_windows = X_test.shape[0]
    print(f"X_test_starter : {X_test.shape}")

    print("Scoring all windows (only needs to run once)")
    probs = score_all_windows(session, input_name, X_test, args.batch_size)
    flags = probs >= threshold
    print(f"Flagged windows : {flags.sum():,} / {n_windows:,} ({flags.mean() * 100:.2f}%)")

    run_start, run_len = longest_run(flags)
    if run_len == 0:
        raise RuntimeError("No window crossed the anomaly threshold anywhere in X_test_starter")
    run_end = run_start + run_len
    print(f"Longest flagged run : {run_len} consecutive windows, starting at window {run_start:,}")

    print("Reconstructing raw (pre-scaling) telemetry from the windowed array")
    scaled_series = np.concatenate([X_test[:, 0, :], X_test[-1, 1:, :]], axis=0)  # (n_windows + window_size - 1, n_features)
    raw_series = scaler.inverse_transform(scaled_series).astype(np.float32)
    n_timesteps = raw_series.shape[0]

    seg_start = max(0, run_start - args.context_before)
    seg_end = min(n_timesteps, run_end + window_size - 1 + args.context_after)
    demo_raw = raw_series[seg_start:seg_end]

    n_demo_windows = demo_raw.shape[0] - window_size + 1
    demo_flags = flags[seg_start: seg_start + n_demo_windows]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "demo_telemetry_raw.npy", demo_raw)

    meta = {
        "starter_cols": config["starter_cols"],
        "window_size": window_size,
        "threshold": threshold,
        "segment_length_timesteps": int(demo_raw.shape[0]),
        "n_demo_windows": int(n_demo_windows),
        "source_seg_start_in_X_test": int(seg_start),
        "flagged_run_length_windows": int(run_len),
        "offline_predicted_flags": demo_flags.astype(bool).tolist(),
        "offline_flagged_window_count": int(demo_flags.sum()),
    }
    with open(args.out_dir / "demo_telemetry_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDemo segment saved to {args.out_dir}/ :")
    print(f"  demo_telemetry_raw.npy   ({demo_raw.nbytes / 1e3:.1f} KB, {demo_raw.shape[0]} raw timesteps)")
    print(f"  demo_telemetry_meta.json")
    print(f"\nReplay will produce {n_demo_windows} windows, {int(demo_flags.sum())} flagged offline (threshold={threshold:.2f})")
    print(f"Copy the models/demo/ folder to the Jetson via USB, then run deployment/telemetry_replay.py")


if __name__ == "__main__":
    main()