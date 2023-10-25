# Rendering scripts

## Installation

```bash
# Download Blender for Linux
wget https://mirrors.ocf.berkeley.edu/blender/release/Blender3.6/blender-3.6.5-linux-x64.tar.xz
# Extract the downloaded Blender archive to /opt/
tar -xvf blender-3.6.5-linux-x64.tar.xz -C /opt/
# Install BlenderProc
pip install blenderproc==2.6.1
# Install required dependencies
apt update && apt install -y libsm6 libglfw3-dev
```

## Render for evaluation

In our evaluation, we render both ground-truth and generated meshes, capturing 24 views around the 3D shape from fixed viewpoints - 12 views at 30° elevation and 12 views at 0° elevation.


```bash
# Render ground-truth meshes
python launch_render_eval.py --DATA_DIR ./examples/objaverse/
# Render One-2-3-45's generated meshes
python launch_render_eval.py --DATA_DIR ./examples/ours/
```
