## Prerequisites

- Python 3.9+
- OS: Linux (recommended) / MacOS / Windows
- For optimal performance: Linux with CUDA-compatible GPU

### System Compatibility Matrix

| OS      | GPU Device | GPU Simulation | CPU Simulation | Interactive Viewer | Headless Rendering |
|---------|------------|----------------|----------------|-------------------|-------------------|
| Linux   | Nvidia     | ✅             | ✅             | ✅               | ✅               |
|         | AMD        | ✅             | ✅             | ✅               | ✅               |
|         | Intel      | ✅             | ✅             | ✅               | ✅               |
| Windows | Nvidia     | ✅             | ✅             | ❌               | ❌               |
|         | AMD        | ✅             | ✅             | ❌               | ❌               |
|         | Intel      | ✅             | ✅             | ❌               | ❌               |
| MacOS   | Apple Silicon | ✅          | ✅             | ✅               | ✅               |


# 🎨 Graphics Driver and CUDA Installation

### 🚮 Removing Existing Graphics Drivers

```bash
sudo apt --purge remove *nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```

### 1️⃣ Installing Graphics Driver

1. Verify and install the driver:<br>
    [Check GPU Driver and CUDA Version Compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4)<br>
   ![Check GPU Driver and CUDA Version Compatibility](https://github.com/user-attachments/assets/70968a52-31c0-415a-a21b-7d6ecdf336b1)
    ```bash
    ubuntu-drivers devices
    sudo apt-get install nvidia-driver-<version number>
    sudo apt-get install dkms nvidia-modprobe -y
    sudo apt-get update
    sudo apt-get upgrade
    sudo reboot now
    ```

3. Verify the installation:

    ```bash
    nvidia-smi
    ```

### 2️⃣ Installing CUDA (Recommended Versions: 12.4)

Refer to the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) to download the runfile, grant execute permissions, and install CUDA.

```bash
chmod 777 <runfile>
nvcc -V  # Verify installation
```

### 3️⃣ Installing cuDNN

1. [Verifying cuDNN Version Compatibility](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).<br>
   ![Verifying cuDNN Version Compatibility](https://github.com/user-attachments/assets/b7e0b101-8f0e-4fdc-9d74-822e3ade1fc3)
3. Download the deb file from the [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) and install it.
4. If needed, use the following commands to remove CUDA and cuDNN source lists.

    ```bash
    sudo rm /etc/apt/sources.list.d/cuda*
    sudo rm /etc/apt/sources.list.d/cudnn*
    ```

# 🔥 Verifying PyTorch and CUDA Installation

[Install using the CUDA-compatible PyTorch installation guide](https://pytorch.org/get-started/locally/)
Run the following Python code to verify CUDA and cuDNN settings:

```python
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.backends.cudnn.enabled)
print(torch.backends.cudnn.version())
```


## Basic Installation

1. Install PyTorch following the [official instructions](https://pytorch.org)
2. Install Genesis:
   ```bash
   pip install genesis-world
   ```

**Note**: If using CUDA, ensure appropriate nvidia-driver is installed.

## Motion Planning
Genesis includes OMPL motion planning functionalities with an intuitive API:
1. [Download pre-compiled OMPL wheel](https://github.com/ompl/ompl/releases/tag/prerelease)
2. Install using pip


# ⚙️ Optional Settings

### 📅 System Update

```bash
sudo apt-get update
sudo apt-get upgrade
```

### ⌨️ Setting up Korean Keyboard

Refer to [this guide](https://shanepark.tistory.com/231) for setting up the Korean keyboard.

### 🐍 Installing pip

```bash
sudo apt-get install python3-pip -y
```

### 💻 Installing Additional Programs

- [GitHub Desktop](https://github.com/shiftkey/desktop/releases/)
- [TeamViewer](https://www.teamviewer.com/ko/download/linux/)
- [VSCode](https://code.visualstudio.com/download)

```bash
sudo apt install barrier -y  # KVM switch software
sudo apt-get install terminator -y  # Convenient terminal
```