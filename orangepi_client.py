#!/usr/bin/env python3
"""
主机端 Orange Pi 客户端
与 Orange Pi 桥接节点通信，支持：
1. 音频录制和唤醒词检测
2. 图像采集
3. ROS2 设备控制
"""

import zmq
import json
import base64
from typing import Optional, Dict, Any


class OrangePiClient:
    """Orange Pi 客户端"""

    def __init__(self, host: str = "192.168.10.55", port: int = 5556):
        """
        初始化客户端

        Args:
            host: Orange Pi IP 地址
            port: ZeroMQ 端口
        """
        self.host = host
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.timeout = 10000  # 10 秒超时
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)

        print(f"[OrangePiClient] 已连接: {host}:{port}")

    def _send_request(self, request: Dict) -> Dict:
        """
        发送请求并接收响应

        Args:
            request: 请求字典

        Returns:
            响应字典
        """
        try:
            self.socket.send_string(json.dumps(request))
            response = self.socket.recv_string()
            return json.loads(response)
        except zmq.error.Again:
            return {"status": "error", "message": "请求超时"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_status(self) -> Dict:
        """
        获取 Orange Pi 状态

        Returns:
            状态信息，包括传感器数据
        """
        return self._send_request({"action": "status"})

    def record_audio(self, duration: int = 5) -> Optional[bytes]:
        """
        录音

        Args:
            duration: 录音时长（秒）

        Returns:
            WAV 格式的音频数据，失败返回 None
        """
        response = self._send_request({
            "action": "record",
            "duration": duration
        })

        if response.get("status") == "ok" and "audio" in response:
            return base64.b64decode(response["audio"])
        return None

    def detect_wake_word(self, listen_duration: int = 3) -> Optional[Dict]:
        """
        检测唤醒词

        Args:
            listen_duration: 监听唤醒词的时长（秒）

        Returns:
            {
                "detected": bool,  # 是否检测到唤醒词
                "audio": bytes,   # 命令音频（如果检测到）
                "format": str,    # 音频格式
                "sample_rate": int  # 采样率
            }
            未检测到返回 None
        """
        response = self._send_request({
            "action": "detect_wake_word",
            "duration": listen_duration
        })

        if response.get("status") == "ok":
            detected = response.get("detected", False)
            if detected and "audio" in response:
                return {
                    "detected": True,
                    "audio": base64.b64decode(response["audio"]),
                    "format": response.get("format", "wav"),
                    "sample_rate": response.get("sample_rate", 16000)
                }
            elif not detected:
                return {"detected": False}

        return None

    def capture_image(self) -> Optional[bytes]:
        """
        采集图像

        Returns:
            JPEG 格式的图像数据，失败返回 None
        """
        response = self._send_request({"action": "capture_image"})

        if response.get("status") == "ok" and "image" in response:
            return base64.b64decode(response["image"])
        return None

    def light_on(self) -> Dict:
        """
        开灯

        Returns:
            响应结果
        """
        return self._send_request({"action": "light_on"})

    def light_off(self) -> Dict:
        """
        关灯

        Returns:
            响应结果
        """
        return self._send_request({"action": "light_off"})

    def arm_move(self, position: Dict[str, float], speed: int = 50) -> Dict:
        """
        移动机械臂

        Args:
            position: 位置坐标 {"x": 0.1, "y": 0.2, "z": 0.3}
            speed: 移动速度

        Returns:
            响应结果
        """
        return self._send_request({
            "action": "arm_move",
            "position": position,
            "speed": speed
        })

    def close(self):
        """关闭连接"""
        self.socket.close()
        self.context.term()


# ==================== 测试 ====================
if __name__ == "__main__":
    import time

    client = OrangePiClient()

    print("测试 Orange Pi 连接...")

    # 测试状态
    print("\n1. 获取状态...")
    status = client.get_status()
    print(f"   状态: {status}")

    # 测试唤醒词检测（会阻塞 3 秒）
    print("\n2. 测试唤醒词检测（请说话）...")
    result = client.detect_wake_word(listen_duration=3)
    if result:
        if result.get("detected"):
            print(f"   检测到唤醒词！音频大小: {len(result.get('audio', b''))} bytes")
        else:
            print("   未检测到唤醒词")

    # 测试录音
    print("\n3. 测试录音...")
    audio = client.record_audio(duration=2)
    if audio:
        print(f"   录音成功: {len(audio)} bytes")
    else:
        print("   录音失败")

    # 测试图像采集
    print("\n4. 测试图像采集...")
    image = client.capture_image()
    if image:
        print(f"   图像采集成功: {len(image)} bytes")
    else:
        print("   图像采集失败")

    client.close()
    print("\n测试完成")
