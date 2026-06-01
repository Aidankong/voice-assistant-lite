#!/bin/bash
# 部署脚本 - 将 Orange Pi 桥接程序部署到香橙派

ORANGE_PI_IP="192.168.10.55"
ORANGE_PI_USER="orangepi"
REMOTE_DIR="~/jarvis"

echo "======================================="
echo "部署 Jarvis 到 Orange Pi"
echo "======================================="
echo "IP: $ORANGE_PI_IP"
echo "用户: $ORANGE_PI_USER"
echo "目标目录: $REMOTE_DIR"
echo ""

# 检查 SSH 连接
echo "检查 SSH 连接..."
ssh -o ConnectTimeout=5 ${ORANGE_PI_USER}@${ORANGE_PI_IP} "echo '连接成功'" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ SSH 连接失败！"
    echo "请确保："
    echo "  1. Orange Pi 已开机"
    echo "  2. 网络连接正常 (ping $ORANGE_PI_IP)"
    echo "  3. SSH 密钥已配置或使用密码登录"
    exit 1
fi
echo "✓ SSH 连接正常"
echo ""

# 在 Orange Pi 上创建目录
echo "创建远程目录..."
ssh ${ORANGE_PI_USER}@${ORANGE_PI_IP} "mkdir -p $REMOTE_DIR"
echo ""

# 上传桥接程序
echo "上传 orangepi_bridge.py..."
scp orangepi_bridge.py ${ORANGE_PI_USER}@${ORANGE_PI_IP}:${REMOTE_DIR}/
echo ""

# 检查 Orange Pi 上的依赖
echo "检查依赖..."
ssh ${ORANGE_PI_USER}@${ORANGE_PI_IP} << 'ENDSSH'
cd ~/jarvis

echo "检查 Python 包..."
python3 -c "import zmq; print('✓ zmq')" 2>/dev/null || echo "✗ 需要安装 pyzmq"
python3 -c "import rclpy; print('✓ rclpy')" 2>/dev/null || echo "✗ 需要安装 rclpy"
python3 -c "import numpy; print('✓ numpy')" 2>/dev/null || echo "✗ 需要安装 numpy"

echo ""
echo "检查音频设备..."
arecord -l 2>/dev/null | grep -E "^card" || echo "⚠ 未找到录音设备"
echo ""
echo "检查摄像头..."
ls /dev/video* 2>/dev/null || echo "⚠ 未找到摄像头设备"
ENDSSH

echo ""
echo "======================================="
echo "部署完成！"
echo "======================================="
echo ""
echo "在 Orange Pi 上运行："
echo "  ssh ${ORANGE_PI_USER}@${ORANGE_PI_IP}"
echo "  cd ~/jarvis"
echo "  python3 orangepi_bridge.py"
echo ""
echo "如需安装缺失依赖："
echo "  pip3 install pyzmq rclpy numpy"
echo ""
