"""
Run on the Jetson — streams the real telemetry segment from prepare_demo_data.py tick by
tick through the deployed model, exactly like a live onboard/ground feed: one new raw
sensor reading arrives, slides into a rolling window buffer, and once the buffer is full
the model scores it. This is a demo of the deployed pipeline end-to-end, not a benchmark
(see benchmark_inference.py for pure latency numbers).
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description="Replay a real telemetry segment live through the deployed model")
    parser.add_argument("--demo-dir", type=Path, default=Path("models/demo"))
    parser.add_argument("--bundle-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--interval", type=float, default=0.05, help="Seconds between simulated telemetry ticks")
    args = parser.parse_args()

    with open(args.bundle_dir / "deployment_config.json") as f:
        config = json.load(f)
    with open(args.demo_dir / "demo_telemetry_meta.json") as f:
        meta = json.load(f)

    scaler = joblib.load(args.bundle_dir / config["scaler_file"])
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3  # suppress the harmless static-output-shape warning
    session = ort.InferenceSession(str(args.bundle_dir / config["model_file"]), sess_options=session_options,
                                    providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    raw_telemetry = np.load(args.demo_dir / "demo_telemetry_raw.npy")
    window_size = config["window_size"]
    threshold = config["best_threshold"]
    starter_cols = config["starter_cols"]

    print(f"Replaying {raw_telemetry.shape[0]} ticks across {len(starter_cols)} channels")
    print(f"Window size: {window_size}  |  Threshold: {threshold:.2f}  |  Interval: {args.interval}s/tick")
    print(f"Offline scan flagged {meta['offline_flagged_window_count']}/{meta['n_demo_windows']} windows in this segment\n")

    buffer = []
    live_flags = []
    n_ticks_scored = 0
    n_flagged = 0

    # Rolling buffer, tick by tick simulation. Retrieve 64 entries to recreate the input of
    # the model, the buffer keeps growing entries until the 64th arrives. Then it follows the FIFO
    # method so it drops the oldest entry to let the 65th entry to come in
    try:
        for tick, raw_row in enumerate(raw_telemetry):
            buffer.append(raw_row)
            if len(buffer) > window_size:
                buffer.pop(0)

            # Scoring gate
            if len(buffer) == window_size:
                scaled = scaler.transform(np.array(buffer)).astype(np.float32)
                model_input = scaled[np.newaxis, :, :]
                # Calling the GRU model
                logit = session.run(None, {input_name: model_input})[0][0]
                prob = 1 / (1 + np.exp(-logit))
                flag = prob >= threshold

                n_ticks_scored += 1
                n_flagged += int(flag)
                live_flags.append(bool(flag))

                status = "*** ANOMALY ***" if flag else "nominal"
                print(f"[tick {tick:5d}] prob={prob:.4f}  {status}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nReplay stopped early by user")

    print(f"\nReplay summary :")
    print(f"  Ticks streamed        : {tick + 1}")
    print(f"  Windows scored        : {n_ticks_scored}")
    print(f"  Windows flagged live  : {n_flagged}")

    offline_flags = meta["offline_predicted_flags"][:len(live_flags)]
    agreement = sum(a == b for a, b in zip(live_flags, offline_flags)) / max(len(live_flags), 1)
    print(f"  Agreement with the local machine, offline scan : {agreement * 100:.2f}%  ({len(live_flags)} windows compared)")


if __name__ == "__main__":
    main()