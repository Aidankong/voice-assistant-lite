#!/usr/bin/env python3
"""
ROS2 桥接节点 - 香橙派端
接收来自高性能电脑的指令，通过 ROS2 控制设备
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import zmq

class ROS2BridgeNode(Node):
    """ROS2 桥接节点"""

    def __init__(self):
        super().__init__('ros2_bridge_node')

        # ZeroMQ 服务器（接收来自高性能电脑的指令）
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind("tcp://*:5556")

        # ROS2 发布者（控制各设备）
        self.cmd_publishers = {
            "ha": self.create_publisher(String, 'home/command', 10),
            "arm": self.create_publisher(String, 'arm/command', 10),
            "base": self.create_publisher(String, 'base/command', 10),
        }

        # 启动接收线程
        import threading
        self.running = True
        threading.Thread(target=self._receive_loop, daemon=True).start()

        self.get_logger().info("ROS2 Bridge 节点已启动")
        self.get_logger().info("ZeroMQ 监听: *:5556")

    def _receive_loop(self):
        """持续接收来自高性能电脑的指令"""
        while self.running and rclpy.ok():
            try:
                # 接收指令
                request = self.zmq_socket.recv_json(flags=zmq.NOBLOCK)
                self.get_logger().info(f"收到指令: {request}")

                # 处理指令
                result = self._handle_command(request)

                # 发送响应
                self.zmq_socket.send_json(result)

            except zmq.Again:
                time.sleep(0.01)
            except Exception as e:
                self.get_logger().error(f"处理错误: {e}")
                try:
                    self.zmq_socket.send_json({"success": False, "error": str(e)})
                except:
                    pass

    def _handle_command(self, request: dict) -> dict:
        """处理命令"""

        tool = request.get('tool')
        params = request.get('params', {})

        self.get_logger().info(f"工具: {tool}, 参数: {params}")

        # 路由到对应的 ROS2 话题
        if tool == "ha_control":
            return self._route_to_publisher("ha", params)
        elif tool == "arm_control":
            return self._route_to_publisher("arm", params)
        elif tool == "base_control":
            return self._route_to_publisher("base", params)
        elif tool == "sensor_query":
            return self._query_sensors(params)
        else:
            return {"success": False, "error": f"Unknown tool: {tool}"}

    def _route_to_publisher(self, key: str, params: dict) -> dict:
        """路由指令到 ROS2 发布者"""
        try:
            msg = String()
            msg.data = json.dumps(params, ensure_ascii=False)
            self.cmd_publishers[key].publish(msg)
            self.get_logger().info(f"已发布到 {key}: {msg.data}")
            return {"success": True, "message": f"{key} command sent"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _query_sensors(self, params: dict) -> dict:
        """查询传感器状态"""
        # 模拟传感器数据
        return {
            "success": True,
            "data": {
                "temperature": 25.5,
                "humidity": 60,
                "timestamp": time.time()
            }
        }


def main(args=None):
    """主函数"""
    rclpy.init(args=args)

    node = ROS2BridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
