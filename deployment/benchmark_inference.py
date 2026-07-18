"""
Runs the exported GRU anomaly detector through ONNX Runtime on-device and reports
inference latency. Meant to run directly on the Jetson against the deployment bundle
copied over via USB (see docs/jetson_setup.md)

Uses synthetic random telemetry windows rather than the real dataset: the deployment
bundle is self-contained on purpose (no raw dataset shipped to the device), and latency
only depends on tensor shape, not values. Correctness against real data was already
verified against the PyTorch model on the dev laptop (notebook 05).
"""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort


def benchmark_latency(session, input_name, window, n_warmup, n_runs):
    for _ in range(n_warmup):
        session.run(None, {input_name: window})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: window})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
    return np.array(latencies)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the GRU anomaly detector (ONNX) on this device")
    parser.add_argument("--bundle-dir", type=Path, default=Path("models/onnx"),
                         help="Folder containing the deployment bundle (onnx model + scaler + config)")
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--n-runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.bundle_dir / "deployment_config.json") as f:
        config = json.load(f)

    scaler = joblib.load(args.bundle_dir / config["scaler_file"])
    session = ort.InferenceSession(str(args.bundle_dir / config["model_file"]), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    print(f"Platform  : {platform.platform()}")
    print(f"Processor : {platform.processor() or platform.machine()}")
    print(f"Providers : {session.get_providers()}")
    print(f"Model     : {config['model_file']}")
    print(f"Threshold : {config['best_threshold']:.4f}")

    window_size = config["window_size"]
    n_features = config["n_features"]

    rng = np.random.default_rng(args.seed)
    raw_window = rng.standard_normal((window_size, n_features)).astype(np.float32)
    scaled_window = scaler.transform(raw_window).astype(np.float32)
    model_input = scaled_window[np.newaxis, :, :]  # (1, window_size, n_features)

    latencies = benchmark_latency(session, input_name, model_input, args.n_warmup, args.n_runs)

    print(f"\nSingle-window inference latency ({args.n_runs} runs, {args.n_warmup} warmup) :")
    print(f"  Mean : {latencies.mean():.3f} ms")
    print(f"  P50  : {np.percentile(latencies, 50):.3f} ms")
    print(f"  P95  : {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99  : {np.percentile(latencies, 99):.3f} ms")

    logit = session.run(None, {input_name: model_input})[0][0]
    prob = 1 / (1 + np.exp(-logit))
    is_anomaly = prob >= config["best_threshold"]
    print(f"\nSample prediction on synthetic random noise (sanity check, not a real signal) :")
    print(f"  Probability : {prob:.4f}  ->  {'ANOMALY' if is_anomaly else 'normal'} (threshold={config['best_threshold']:.2f})")


if __name__ == "__main__":
    main()