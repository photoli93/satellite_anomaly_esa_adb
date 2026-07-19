# Jetson Orin Nano Setup - Deployment

## Hardware & software confirmed on this board

- **Board**: Jetson Orin Nano Super Developer Kit (8GB)
- **JetPack**: 7.2 (`nvidia-jetpack-runtime = 7.2-b184`)
- **Jetson Linux (L4T)**: 39.2
- **CUDA**: 13.2.1
- **TensorRT**: 10.16.2
- **Base OS**: Ubuntu 24.04 (aarch64), kernel 6.8
- Flashed via the unified USB ISO image (JetPack 7.2 dropped the SD-card image path for the Orin Nano dev kit)

## Deployment path: CPU-only

`onnxruntime-gpu` has no prebuilt pip wheel for aarch64 + CUDA 13 + sm_87 (Orin's compute capability) as of JetPack 7.2, PyPI only ships x86_64/Windows wheels under that name. Building it from source with the TensorRT execution provider takes roughly 4 hours and needs ~30GB disk + 16GB swap on an 8GB board and the only available build path ([straga/jetson-jp7-onnxruntime](https://github.com/straga/jetson-jp7-onnxruntime)) is community-maintained, not official, JetPack 7.2 is too new for NVIDIA/Microsoft to have published one yet.

Decision for this milestone: deploy plain CPU-only ONNX Runtime (aarch64 wheel exists and installs normally via pip) to get a real, working, end-to-end demo on the actual hardware. GPU/TensorRT execution is a stretch goal for later.

## 1. First boot

If you haven't already gone through Ubuntu's first-run wizard after flashing:
- Create a local user, set hostname, connect to Wi-Fi or Ethernet
- Enable SSH for headless access later: `sudo systemctl enable --now ssh`
- Confirm you can reach the board over the network: `ip a` on the Jetson, then `ssh <user>@<jetson-ip>` from your laptop

(View the [Quick start guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html))

## 2. Confirm the software stack

```bash
sudo apt show nvidia-jetpack        # should show 7.2-b184 or later
cat /etc/nv_tegra_release           # L4T build info
python3 --version                   # Ubuntu 24.04 default (check the exact version, do not assume)
nproc                               # CPU core count, useful context for the latency numbers later
free -h                             # confirm the 8GB RAM budget
```

## 3. Python environment

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip

cd ~
python3 -m venv sat-anom-jetson-env
source sat-anom-jetson-env/bin/activate
```

## 4. Install dependencies

Run [`deployment/install_jetson.sh`](../deployment/install_jetson.sh) (copy the `deployment/` folder to the Jetson via USB):

```bash
bash deployment/install_jetson.sh
```

This installs `onnxruntime` (CPU execution provider), `numpy`, `scikit-learn` and `joblib` pinned to the versions used when the model and scaler were produced on the dev laptop (see [`deployment/requirements.txt`](../deployment/requirements.txt)) — this matters because the scaler is a `joblib`-pickled `StandardScaler` object, and loading it with a mismatched `scikit-learn` version can throw compatibility warnings or, in the worst case, silently deserialize incorrectly.

## 5. Transfer the deployment bundle

Copy the entire `models/onnx/` folder from the dev laptop to the Jetson via USB drive — all four files together (`gru_anomaly_detector.onnx`, `gru_anomaly_detector.onnx.data`, `scaler_starter_channels.pkl`, `deployment_config.json`). The `.onnx` and `.onnx.data` files are a pair; ONNX Runtime looks for the `.data` file by relative name next to the `.onnx` file, so they must stay in the same directory.

```bash
# on the Jetson, after plugging in the USB drive
mkdir -p ~/sat-anomaly/models/onnx
cp /media/<user>/<usb-drive>/onnx/* ~/sat-anomaly/models/onnx/
ls -la ~/sat-anomaly/models/onnx/
```

## 6. Run the latency benchmark

```bash
source ~/sat-anom-jetson-env/bin/activate
python3 deployment/benchmark_inference.py --bundle-dir ~/sat-anomaly/models/onnx
```

This loads the ONNX model + scaler from the bundle, runs a warmup, then times single-window inference the same way notebook 05 did in local. It's giving a genuine Jetson-CPU-vs-laptop-CPU comparison (laptop reference: mean 0.088 ms, P99 0.154 ms)

## 7. Stretch goal: GPU / TensorRT execution provider (not required for Milestone 2)

If there's time later: build `onnxruntime` from source with `--use_tensorrt` following [straga/jetson-jp7-onnxruntime](https://github.com/straga/jetson-jp7-onnxruntime) (budget ~4 hours, ~30GB free disk, 16GB swap). Given the model is tiny (15,937 parameters, 31KB graph) and the CPU latency is already sub-millisecond on a laptop, GPU/TensorRT is unlikely to change the deployment story materially — it's a "nice to demonstrate the full edge stack" item, not a performance necessity for this model.

## Ressources used :

[Deploying AI Models A Step-by-Step Guide](https://zenvanriel.com/ai-engineer-blog/deploying-ai-models-step-by-step-guide/) \
[Deploying AI/ML Models to Production: A Step-by-Step Guide](https://medium.com/@ann.vettoor/deploying-ai-ml-models-to-production-a-step-by-step-guide-3a4df25b74f0) \
[A Beginner-Friendly Guide to Deployment and Production in AI Systems](https://medium.com/tech-ai-made-easy/a-beginner-friendly-guide-to-deployment-and-production-in-ai-systems-4ceb24c1f1ec) \
[Jetson AI Lab tutorials](https://www.jetson-ai-lab.com/tutorials/genai-on-jetson-llms-vlms/) \
[Bringing AI to the Edge: Deploying Machine Learning Models on Edge Devices](https://medium.com/@rijuldahiya/bringing-ai-to-the-edge-deploying-machine-learning-models-on-edge-devices-de7c39438f2e) \
[]() \
[]() \
[]()
