#!/usr/bin/env python3
"""
测试 ROS2 桥接连接
"""

import zmq
import json

def test_ros2_bridge():
    """测试连接到香橙派的 ROS2 桥接节点"""

    print("="*50)
    print("测试 ROS2 桥接连接")
    print("="*50)

    # 创建 ZeroMQ 客户端
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时

    # 连接到香橙派
    orange_pi_ip = "192.168.10.55:5556"
    print(f"\n连接到 {orange_pi_ip}...")

    try:
        socket.connect(f"tcp://{orange_pi_ip}")
        print("✓ 连接成功！")

        # 测试 1: 打开灯光
        print("\n测试 1: 打开客厅灯光")
        request = {
            "tool": "ha_control",
            "params": {
                "entity_id": "light.living_room",
                "action": "on"
            }
        }
        socket.send_json(request)
        response = socket.recv_json()
        print(f"发送: {request}")
        print(f"响应: {response}")

        # 测试 2: 查询传感器
        print("\n测试 2: 查询传感器")
        request = {
            "tool": "sensor_query",
            "params": {}
        }
        socket.send_json(request)
        response = socket.recv_json()
        print(f"发送: {request}")
        print(f"响应: {response}")

        # 测试 3: 机械臂控制
        print("\n测试 3: 机械臂移动")
        request = {
            "tool": "arm_control",
            "params": {
                "action": "move",
                "position": {"x": 100, "y": 200, "z": 50}
            }
        }
        socket.send_json(request)
        response = socket.recv_json()
        print(f"发送: {request}")
        print(f"响应: {response}")

        print("\n" + "="*50)
        print("✓ 所有测试完成！")
        print("="*50)

    except zmq.error.Again:
        print("✗ 连接超时")
    except Exception as e:
        print(f"✗ 错误: {e}")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    test_ros2_bridge()
