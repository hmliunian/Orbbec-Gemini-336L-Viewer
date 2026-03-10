# Orbbec Gemini 336L Viewer

基于 pyorbbecsdk 的 Orbbec Gemini 336L 相机可视化工具，支持彩色/深度流预览、截图、录像和点云导出。

## 环境准备

```bash
uv sync
```

pyorbbecsdk 的 PyPI 包在 Linux 上有兼容问题，需从源码编译安装：

```bash
git clone https://github.com/orbbec/pyorbbecsdk.git /tmp/pyorbbecsdk
cd /tmp/pyorbbecsdk && git checkout main
mkdir build && cd build
cmake -Dpybind11_DIR=$(pybind11-config --cmakedir) ..
make -j$(nproc)
```

将编译产物复制到 venv：

```bash
SITE=".venv/lib/python3.11/site-packages"
cp /tmp/pyorbbecsdk/build/pyorbbecsdk.cpython-311-x86_64-linux-gnu.so $SITE/
cp /tmp/pyorbbecsdk/sdk/lib/linux_x64/*.so* $SITE/
```

安装 udev 规则（需要 sudo）：

```bash
sudo cp /tmp/pyorbbecsdk/99-obsensor-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## 运行

```bash
LD_LIBRARY_PATH=.venv/lib/python3.11/site-packages:$LD_LIBRARY_PATH uv run python main.py
```

## 功能

- 实时预览彩色流和深度流
- 截图保存（PNG + 原始深度 NPY）
- 录像（AVI）
- 点云导出（PLY，需要 open3d）

输出文件保存在 `output/` 目录。
