# Jarvis 分布式语音助手 - 部署指南

## 架构概述

```
┌─────────────────┐         ┌─────────────────┐
│   Orange Pi     │         │   主机 (PC)     │
│   (机器人端)     │         │   (高性能端)     │
├─────────────────┤         ├─────────────────┤
│ • 麦克风录音     │────────▶│ • Whisper 识别  │
│ • 唤醒词检测     │ ZeroMQ  │ • Qwen2.5 LLM   │
│ • 图像采集(预留) │◀────────│ • TTS 播放      │
│ • ROS2 控制     │         │ • 意图识别      │
└─────────────────┘         └─────────────────┘
```

## 部署步骤

### 1. Orange Pi 端配置

#### 安装依赖
```bash
# SSH 登录 Orange Pi
ssh orangepi@192.168.10.55

# 安装 Python 依赖
pip3 install pyzmq rclpy numpy

# 或使用系统包管理器
sudo apt install python3-zmq python3-numpy
```

#### 部署桥接程序
```bash
# 在主机上运行
cd /home/luxrobot/voice-assistant-lite
./deploy_to_orangepi.sh
```

#### 启动桥接程序
```bash
# 在 Orange Pi 上
ssh orangepi@192.168.10.55
cd ~/jarvis
python3 orangepi_bridge.py
```

### 2. 主机端配置

#### 确保依赖已安装
```bash
pip install ollama faster-whisper edge-tts pyzmq numpy
```

#### 运行语音助手
```bash
cd /home/luxrobot/voice-assistant-lite

# 文本模式测试
python3 voice_assistant.py

# 语音模式测试
python3 voice_assistant.py --voice
```

### 3. 测试连接

```bash
# 测试 Orange Pi 连接
python3 -c "from orangepi_client import OrangePiClient; c = OrangePiClient(); print(c.get_status())"
```

## 文件说明

| 文件 | 说明 | 运行位置 |
|------|------|----------|
| `orangepi_bridge.py` | Orange Pi 桥接节点 | Orange Pi |
| `orangepi_client.py` | 主机端客户端 | 主机 |
| `voice_assistant.py` | 语音助手主程序 | 主机 |
| `voice_io.py` | 语音 I/O 模块 | 主机 |
| `deploy_to_orangepi.sh` | 部署脚本 | 主机 |

## 功能特性

### 当前实现
- ✅ ZeroMQ 通信框架
- ✅ Orange Pi 音频录制
- ✅ Orange Pi 唤醒词检测（基础版）
- ✅ 主机端 Whisper 识别
- ✅ 主机端 Qwen2.5 LLM 处理
- ✅ ROS2 设备控制接口
- ✅ 图像采集接口（预留）

### 待优化
- ⏳ 唤醒词检测升级（Porcupine/Snowboy）
- ⏳ 图像传输和处理
- ⏳ 视觉功能集成

## 故障排查

### Orange Pi 无法连接
```bash
# 检查网络
ping 192.168.10.55

# 检查 SSH
ssh orangepi@192.168.10.55

# 检查防火墙
telnet 192.168.10.55 5556
```

### 音频设备问题
```bash
# 在 Orange Pi 上检查录音设备
arecord -l

# 测试录音
arecord -d 3 -f S16_LE -c 1 -r 16000 /tmp/test.wav
aplay /tmp/test.wav
```

### 摄像头问题
```bash
# 检查摄像头设备
ls /dev/video*

# 测试采集
fswebcam test.jpg
```
