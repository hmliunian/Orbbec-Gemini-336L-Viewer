export LD_LIBRARY_PATH := ".venv/lib/python3.11/site-packages:" + env("LD_LIBRARY_PATH", "")

# 启动 GUI
run:
    uv run python main.py

# 安装依赖
setup:
    uv sync
    uv pip install pybind11

# 从源码编译安装 pyorbbecsdk (Linux)
build-sdk:
    #!/usr/bin/env bash
    set -euo pipefail
    git clone https://github.com/orbbec/pyorbbecsdk.git /tmp/pyorbbecsdk || true
    cd /tmp/pyorbbecsdk && git checkout main
    mkdir -p build && cd build
    cmake -Dpybind11_DIR=$(pybind11-config --cmakedir) ..
    make -j$(nproc)
    SITE=".venv/lib/python3.11/site-packages"
    cp /tmp/pyorbbecsdk/build/pyorbbecsdk.cpython-311-x86_64-linux-gnu.so {{justfile_directory()}}/$SITE/
    cp /tmp/pyorbbecsdk/sdk/lib/linux_x64/*.so* {{justfile_directory()}}/$SITE/
    echo "SDK installed to $SITE"

# 查看 PLY 点云文件
view-ply file:
    uv run python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('{{file}}'); o3d.visualization.draw_geometries([pcd])"

# 安装 udev 规则 (需要 sudo)
install-udev:
    sudo cp /tmp/pyorbbecsdk/99-obsensor-libusb.rules /etc/udev/rules.d/
    sudo udevadm control --reload-rules && sudo udevadm trigger
