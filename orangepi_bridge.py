#!/usr/bin/env python3
"""
Orange Pi 多功能桥接节点
功能：
1. 音频录制和唤醒词检测
2. 图像采集（预留）
3. ROS2 设备控制
4. 与主机的 ZeroMQ 通信
"""

import asyncio
import zmq
import zmq.asyncio
import json
import time
import os
import sys
import subprocess
import tempfile
import wave
import numpy as np
from typing import Optional, Dict, Any

# 尝试导入 ROS2，如果失败则继续运行（无 ROS2 模式）
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("[WARNING] ROS2 not available, running in non-ROS2 mode")


# ==================== 配置 ====================
CONFIG = {
    # ZeroMQ
    "zmq_host": "0.0.0.0",  # 监听所有接口
    "zmq_port": 5556,

    # 音频
    "audio_sample_rate": 16000,
    "audio_channels": 1,  # 输出单声道
    "audio_record_channels": 2,  # Orange Pi 需要立体声录制
    "audio_duration": 5,  # 命令录音时长（秒）
    "wake_word": "jarvis",
    "wake_threshold": 0.3,  # 唤醒词相似度阈值

    # ROS2
    "ros_topics": {
        "light_cmd": "/light/command",
        "arm_cmd": "/arm/command",
        "sensor_data": "/sensor/data",
    }
}


# ==================== 轻量级唤醒词检测 ====================
class WakeWordDetector:
    """简单的基于能量的唤醒词检测（可替换为更复杂的模型）"""

    def __init__(self, wake_word: str = "jarvis", threshold: float = 0.3):
        self.wake_word = wake_word.lower()
        self.threshold = threshold
        # 这里可以用 porcupine 或其他轻量级 KWS 模型

    def detect(self, audio_data: bytes) -> bool:
        """
        检测音频中的唤醒词
        简化版：检测是否有足够的语音活动
        TODO: 可以替换为真正的关键词检测模型
        """
        # 将音频转换为 numpy 数组
        samples = np.frombuffer(audio_data, dtype=np.int16)
        energy = np.mean(np.abs(samples))

        # 简单的能量阈值检测（实际应该用更智能的方法）
        # 这里返回 True 模拟检测到唤醒词
        # 在实际部署时，可以用 porcupine、snowboy 或其他 KWS
        return energy > 1000  # 临时阈值，表示有声音


# ==================== 音频录制模块 ====================
class AudioRecorder:
    """音频录制模块"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.record_channels = CONFIG["audio_record_channels"]  # 录音通道数
        self.device = self._find_microphone()

    def _find_microphone(self) -> str:
        """查找可用的麦克风设备"""
        try:
            result = subprocess.run(
                ["arecord", "-l"],
                capture_output=True,
                text=True
            )
            # 解析输出找到第一个录音设备
            for line in result.stdout.split('\n'):
                if 'card' in line and 'device' not in line.lower():
                    # 提取设备号
                    parts = line.split(':')
                    if len(parts) >= 2:
                        card_num = parts[0].split()[-1]
                        return f"hw:{card_num},0"

            # 默认使用第一个设备
            return "default"
        except Exception as e:
            print(f"[AudioRecorder] 查找麦克风失败: {e}")
            return "default"

    def _stereo_to_mono(self, wav_path: str) -> bytes:
        """将立体声 WAV 文件转换为单声道字节"""
        with wave.open(wav_path, 'rb') as wav_file:
            # 读取立体声参数
            frames = wav_file.getnframes()
            # 读取原始音频数据
            stereo_data = wav_file.readframes(frames)

        # 转换为 numpy 数组（立体声）
        stereo = np.frombuffer(stereo_data, dtype=np.int16)
        # 重塑为 (n, 2)
        stereo = stereo.reshape(-1, 2)
        # 平均两个声道
        mono = stereo.mean(axis=1).astype(np.int16)
        return mono.tobytes()

    def record(self, duration: int) -> Optional[bytes]:
        """
        录音指定时长

        Args:
            duration: 录音时长（秒）

        Returns:
            WAV 格式的音频数据（单声道）
        """
        try:
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False
            )
            temp_path = temp_file.name
            temp_file.close()

            # 先用立体声录制
            cmd = [
                "arecord",
                "-q",
                "-d", str(duration),
                "-f", "S16_LE",
                "-c", str(self.record_channels),  # 使用立体声录制
                "-r", str(self.sample_rate),
                "-D", self.device,
                temp_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=duration + 5
            )

            if result.returncode == 0:
                # 如果录制的是立体声，需要转换为单声道
                if self.record_channels == 2 and self.channels == 1:
                    # 使用 wave 库正确读取和转换
                    mono_data = self._stereo_to_mono(temp_path)
                    os.unlink(temp_path)

                    # 封装为单声道 WAV
                    temp_out = tempfile.NamedTemporaryFile(
                        suffix='.wav',
                        delete=False
                    )
                    temp_out_path = temp_out.name
                    temp_out.close()

                    with wave.open(temp_out_path, 'wb') as wav:
                        wav.setnchannels(self.channels)
                        wav.setsampwidth(2)  # 16-bit
                        wav.setframerate(self.sample_rate)
                        wav.writeframes(mono_data)

                    with open(temp_out_path, 'rb') as f:
                        audio_data = f.read()

                    os.unlink(temp_out_path)
                    return audio_data
                else:
                    # 直接返回录制的文件
                    with open(temp_path, 'rb') as f:
                        audio_data = f.read()
                    os.unlink(temp_path)
                    return audio_data
            else:
                print(f"[AudioRecorder] 录音失败: {result.stderr.decode()}")
                os.unlink(temp_path)
                return None

        except Exception as e:
            print(f"[AudioRecorder] 录音异常: {e}")
            return None


# ==================== 图像采集模块（预留） ====================
class CameraCapture:
    """图像采集模块（为未来视觉功能预留）"""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id

    def capture(self) -> Optional[bytes]:
        """
        采集一帧图像

        Returns:
            JPEG 格式的图像数据
        """
        try:
            # 使用 fswebcam 或其他工具
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.jpg',
                delete=False
            )
            temp_path = temp_file.name
            temp_file.close()

            cmd = [
                "fswebcam",
                "-q",
                "--no-banner",
                temp_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=5)

            if result.returncode == 0:
                with open(temp_path, 'rb') as f:
                    image_data = f.read()
                os.unlink(temp_path)
                return image_data
            else:
                print(f"[CameraCapture] 采集失败: {result.stderr.decode()}")
                os.unlink(temp_path)
                return None

        except FileNotFoundError:
            print("[CameraCapture] fswebcam 未安装")
            return None
        except Exception as e:
            print(f"[CameraCapture] 采集异常: {e}")
            return None


# ==================== ROS2 节点 ====================
class OrangePiBridge:
    """Orange Pi 桥接节点"""

    def __init__(self):
        self.sensor_data = {}
        self.light_pub = None
        self.arm_pub = None
        self.sensor_sub = None
        self.ros_node = None

        if HAS_ROS2:
            try:
                rclpy.init()
                self.ros_node = rclpy.create_node('orangepi_bridge')

                # ROS2 发布者
                self.light_pub = self.ros_node.create_publisher(
                    String,
                    CONFIG["ros_topics"]["light_cmd"],
                    10
                )
                self.arm_pub = self.ros_node.create_publisher(
                    String,
                    CONFIG["ros_topics"]["arm_cmd"],
                    10
                )
                self.sensor_sub = self.ros_node.create_subscription(
                    String,
                    CONFIG["ros_topics"]["sensor_data"],
                    self.sensor_callback,
                    10
                )

                print("[ROS2] 节点已初始化")
            except Exception as e:
                print(f"[ROS2] 初始化失败: {e}")
                self.ros_node = None
        else:
            print("[ROS2] 不可用，运行在无 ROS2 模式")

    def sensor_callback(self, msg: String):
        """传感器数据回调"""
        try:
            data = json.loads(msg.data)
            self.sensor_data.update(data)
        except json.JSONDecodeError:
            pass

    def publish_light_cmd(self, state: str):
        """发布灯光控制命令"""
        if self.light_pub:
            msg = String()
            msg.data = json.dumps({"action": "light", "state": state})
            self.light_pub.publish(msg)

    def publish_arm_cmd(self, cmd: Dict[str, Any]):
        """发布机械臂控制命令"""
        if self.arm_pub:
            msg = String()
            msg.data = json.dumps({"action": "arm", **cmd})
            self.arm_pub.publish(msg)

    def get_sensor_data(self) -> Dict:
        """获取传感器数据"""
        return self.sensor_data

    def spin_once(self):
        """执行一次 ROS2 spin"""
        if self.ros_node:
            rclpy.spin_once(self.ros_node, timeout_sec=0.001)

    def shutdown(self):
        """关闭 ROS2"""
        if self.ros_node:
            self.ros_node.destroy_node()
        if HAS_ROS2 and rclpy.ok():
            rclpy.shutdown()


# ==================== ZeroMQ 服务器 ====================
class ZMQServer:
    """ZeroMQ 服务器 - 处理来自主机的请求"""

    def __init__(self, ros_node: OrangePiBridge):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{CONFIG['zmq_host']}:{CONFIG['zmq_port']}")

        self.ros_node = ros_node
        self.audio_recorder = AudioRecorder()
        self.wake_detector = WakeWordDetector(
            wake_word=CONFIG["wake_word"],
            threshold=CONFIG["wake_threshold"]
        )
        self.camera = CameraCapture()

        print(f"[ZMQ] 服务器监听: {CONFIG['zmq_host']}:{CONFIG['zmq_port']}")

    async def handle_request(self, request: Dict) -> Dict:
        """处理客户端请求"""
        action = request.get("action")

        if action == "status":
            # 返回状态信息
            return {
                "status": "ok",
                "ros2_available": HAS_ROS2,
                "sensors": self.ros_node.get_sensor_data()
            }

        elif action == "record":
            # 录音
            duration = request.get("duration", CONFIG["audio_duration"])
            audio_data = self.audio_recorder.record(duration)

            if audio_data:
                # 将音频数据编码为 base64
                import base64
                return {
                    "status": "ok",
                    "audio": base64.b64encode(audio_data).decode(),
                    "format": "wav",
                    "sample_rate": CONFIG["audio_sample_rate"]
                }
            else:
                return {"status": "error", "message": "录音失败"}

        elif action == "detect_wake_word":
            # 检测唤醒词
            duration = request.get("duration", 3)
            audio_data = self.audio_recorder.record(duration)

            if audio_data:
                detected = self.wake_detector.detect(audio_data)

                if detected:
                    # 录制命令音频
                    cmd_audio = self.audio_recorder.record(CONFIG["audio_duration"])
                    import base64
                    return {
                        "status": "ok",
                        "detected": True,
                        "audio": base64.b64encode(cmd_audio).decode() if cmd_audio else None,
                        "format": "wav",
                        "sample_rate": CONFIG["audio_sample_rate"]
                    }
                else:
                    return {"status": "ok", "detected": False}
            else:
                return {"status": "error", "message": "录音失败"}

        elif action == "capture_image":
            # 采集图像
            image_data = self.camera.capture()

            if image_data:
                import base64
                return {
                    "status": "ok",
                    "image": base64.b64encode(image_data).decode(),
                    "format": "jpeg"
                }
            else:
                return {"status": "error", "message": "图像采集失败"}

        elif action == "light_on":
            # 开灯
            self.ros_node.publish_light_cmd("on")
            return {"status": "ok", "message": "灯光已开启"}

        elif action == "light_off":
            # 关灯
            self.ros_node.publish_light_cmd("off")
            return {"status": "ok", "message": "灯光已关闭"}

        elif action == "arm_move":
            # 移动机械臂
            self.ros_node.publish_arm_cmd({
                "position": request.get("position"),
                "speed": request.get("speed", 50)
            })
            return {"status": "ok", "message": "机械臂移动中"}

        else:
            return {"status": "error", "message": "未知命令"}

    async def run(self):
        """运行服务器"""
        while True:
            try:
                # 接收请求
                message = await self.socket.recv_string()
                request = json.loads(message)

                print(f"[ZMQ] 收到请求: {request.get('action')}")

                # 处理 ROS2 回调
                self.ros_node.spin_once()

                # 处理请求
                response = await self.handle_request(request)

                # 发送响应
                await self.socket.send_string(json.dumps(response))

            except json.JSONDecodeError:
                await self.socket.send_string(json.dumps({
                    "status": "error",
                    "message": "无效的 JSON"
                }))
            except Exception as e:
                print(f"[ZMQ] 错误: {e}")
                await self.socket.send_string(json.dumps({
                    "status": "error",
                    "message": str(e)
                }))


# ==================== 主程序 ====================
async def main():
    # 初始化 Orange Pi 桥接节点
    ros_node = OrangePiBridge()

    # 启动 ZMQ 服务器
    zmq_server = ZMQServer(ros_node)

    print("=" * 50)
    print("Orange Pi 桥接节点")
    print("=" * 50)
    print(f"ZMQ: {CONFIG['zmq_host']}:{CONFIG['zmq_port']}")
    print(f"唤醒词: {CONFIG['wake_word']}")
    print(f"ROS2: {'启用' if HAS_ROS2 else '不可用'}")
    if HAS_ROS2:
        print(f"ROS2 话题: {list(CONFIG['ros_topics'].values())}")
    print("=" * 50)

    try:
        await zmq_server.run()
    except KeyboardInterrupt:
        print("\n关闭中...")
    finally:
        ros_node.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
