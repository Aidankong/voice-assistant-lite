"""
语音助手核心类 - 方案B自建轻量框架
核心功能：意图识别 + 工具执行 + 记忆系统
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from ollama import Client
import zmq

# ==================== 数据类 ====================

@dataclass
class Intent:
    """意图识别结果"""
    name: str                    # 意图名称
    confidence: float            # 置信度
    entities: Dict[str, str]     # 提取的实体


@dataclass
class ToolCall:
    """工具调用"""
    name: str
    params: Dict
    result: Optional[Dict] = None


@dataclass
class ConversationContext:
    """对话上下文"""
    session_id: str
    messages: List[Dict] = field(default_factory=list)
    last_intent: Optional[Intent] = None
    last_tool_result: Optional[Dict] = None


# ==================== 核心类 ====================

class VoiceAssistant:
    """轻量级语音助手核心类"""

    def __init__(self, config: dict = None):
        # 默认配置
        self.config = config or {}
        self.ollama_host = self.config.get("ollama_host", "localhost:11434")
        self.model_name = self.config.get("model", "qwen2.5:14b")
        self.orange_pi_addr = self.config.get("orange_pi_addr", "192.168.10.55:5556")

        # 初始化 LLM 客户端
        self.llm = Client(host=self.ollama_host)

        # 初始化 ROS2 桥接（ZeroMQ）
        self._init_ros2_bridge()

        # 工具注册表
        self.tools: Dict[str, Callable] = {}

        # 对话上下文
        self.context: Optional[ConversationContext] = None

        print(f"[VoiceAssistant] 初始化完成，使用模型: {self.model_name}")
        print(f"[VoiceAssistant] ROS2 桥接: {self.orange_pi_addr}")

    def _init_ros2_bridge(self):
        """初始化 ROS2 桥接"""
        try:
            self.zmq_context = zmq.Context()
            self.ros2_socket = self.zmq_context.socket(zmq.REQ)
            self.ros2_socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2秒超时
            print(f"[VoiceAssistant] ZeroMQ 客户端已创建")
        except Exception as e:
            print(f"[VoiceAssistant] ZeroMQ 初始化失败: {e}")
            self.zmq_context = None
            self.ros2_socket = None

    # ==================== 工具注册 ====================

    def register_tool(self, name: str, func: Callable, description: str = ""):
        """注册工具"""
        self.tools[name] = {
            "func": func,
            "description": description
        }
        print(f"[VoiceAssistant] 已注册工具: {name}")

    def list_tools(self) -> str:
        """获取工具列表（用于 Prompt）"""
        descriptions = []
        for name, tool_info in self.tools.items():
            desc = f"- {name}: {tool_info['description']}"
            descriptions.append(desc)
        return "\n".join(descriptions)

    # ==================== 主处理流程 ====================

    async def process(self, text_input: str) -> str:
        """处理文本输入，返回响应"""

        print(f"\n[VoiceAssistant] 收到输入: {text_input}")

        # 1. 意图识别
        intent = await self._classify_intent(text_input)
        print(f"[VoiceAssistant] 意图: {intent.name}, 置信度: {intent.confidence}")

        # 2. 决策：是否需要工具
        tool_result = None
        if intent.name != "chat" and intent.name in self.tools:
            # 执行工具
            tool_call = ToolCall(
                name=intent.name,
                params=intent.entities
            )
            print(f"[VoiceAssistant] 执行工具: {intent.name}")
            tool_result = await self._execute_tool(tool_call)
            print(f"[VoiceAssistant] 工具结果: {tool_result}")

        # 3. 生成回复
        response = await self._generate_response(
            user_input=text_input,
            intent=intent,
            tool_result=tool_result
        )
        print(f"[VoiceAssistant] 回复: {response}")

        return response

    # ==================== 意图分类 ====================

    async def _classify_intent(self, text: str) -> Intent:
        """使用 LLM 进行意图分类"""

        tools_list = self.list_tools()

        prompt = f"""你是语音助手的意图分类器。分析用户输入，提取意图和实体。

用户输入：{text}

可用工具：
{tools_list}

请以 JSON 格式返回，不要有任何其他内容：
{{
    "intent": "工具名称 或 'chat'（如果是闲聊）",
    "confidence": 0.95,
    "entities": {{"key": "value"}}
}}

只返回 JSON，不要解释。"""

        try:
            response = self.llm.generate(
                model=self.model_name,
                prompt=prompt,
                format="json"
            )

            result = json.loads(response['response'])
            return Intent(
                name=result['intent'],
                confidence=result.get('confidence', 0.8),
                entities=result.get('entities', {})
            )
        except Exception as e:
            print(f"[VoiceAssistant] 意图分类失败: {e}")
            # 解析失败，默认为闲聊
            return Intent(name="chat", confidence=0.5, entities={})

    # ==================== 工具执行 ====================

    async def _execute_tool(self, tool_call: ToolCall) -> Dict:
        """执行工具调用"""

        if tool_call.name not in self.tools:
            return {"success": False, "error": f"Unknown tool: {tool_call.name}"}

        try:
            tool_func = self.tools[tool_call.name]["func"]
            result = await tool_func(tool_call.params)
            tool_call.result = result
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================== 响应生成 ====================

    async def _generate_response(
        self,
        user_input: str,
        intent: Intent,
        tool_result: Optional[Dict]
    ) -> str:
        """生成回复"""

        # 构建上下文
        context_parts = [f"用户说：{user_input}"]

        if tool_result:
            if tool_result.get('success'):
                context_parts.append("操作执行成功")
            else:
                context_parts.append(f"操作失败：{tool_result.get('error', '未知错误')}")

        prompt = f"""你是智能家居语音助手，根据以下信息生成简洁、自然的回复。

{chr(10).join(context_parts)}

要求：
1. 回复简洁，控制在20字以内
2. 语气自然、友好
3. 确认执行的操作
4. 如果操作失败，说明原因

只返回回复内容，不要其他文字。"""

        try:
            response = self.llm.generate(model=self.model_name, prompt=prompt)
            return response['response'].strip()
        except Exception as e:
            print(f"[VoiceAssistant] 生成回复失败: {e}")
            return "抱歉，我遇到了一些问题。"

    # ==================== ROS2 工具 ====================

    async def _send_to_ros2(self, tool: str, params: Dict) -> Dict:
        """发送指令到 ROS2 桥接"""
        if not self.ros2_socket:
            return {"success": False, "error": "ROS2 桥接未初始化"}

        try:
            # 连接到香橙派
            self.ros2_socket.connect(f"tcp://{self.orange_pi_addr}")

            # 发送请求
            request = {"tool": tool, "params": params}
            self.ros2_socket.send_json(request)

            # 接收响应
            response = self.ros2_socket.recv_json()

            # 断开连接
            self.ros2_socket.disconnect(f"tcp://{self.orange_pi_addr}")

            return response
        except zmq.error.Again:
            return {"success": False, "error": "ROS2 桥接超时"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def ha_control(self, params: Dict) -> Dict:
        """Home Assistant 设备控制"""
        return await self._send_to_ros2("ha_control", params)

    async def arm_control(self, params: Dict) -> Dict:
        """机械臂控制"""
        return await self._send_to_ros2("arm_control", params)

    async def base_control(self, params: Dict) -> Dict:
        """移动底盘控制"""
        return await self._send_to_ros2("base_control", params)

    async def sensor_query(self, params: Dict) -> Dict:
        """传感器查询"""
        return await self._send_to_ros2("sensor_query", params)

    # ==================== 测试工具 ====================

    async def test_tool(self, params: Dict) -> Dict:
        """测试工具"""
        print(f"[测试工具] 参数: {params}")
        return {"success": True, "message": "测试成功"}


# ==================== 入口 ====================

async def main():
    """主函数 - 测试"""

    # 创建助手实例
    config = {
        "ollama_host": "localhost:11434",
        "model": "qwen2.5:14b",
        "orange_pi_addr": "192.168.10.55:5556"
    }
    assistant = VoiceAssistant(config)

    # 注册 ROS2 工具
    assistant.register_tool(
        "light_on",
        assistant.ha_control,
        "打开灯光（参数：entity_id=设备ID）"
    )

    assistant.register_tool(
        "light_off",
        assistant.ha_control,
        "关闭灯光（参数：entity_id=设备ID）"
    )

    assistant.register_tool(
        "arm_move",
        assistant.arm_control,
        "机械臂移动（参数：position=坐标）"
    )

    assistant.register_tool(
        "sensor_query",
        assistant.sensor_query,
        "查询传感器状态"
    )

    # 交互式测试
    print("\n" + "="*50)
    print("语音助手测试模式（已连接 ROS2）")
    print("输入文本测试，输入 'quit' 退出")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("你: ")

            if user_input.lower() in ['quit', 'exit', '退出']:
                break

            response = await assistant.process(user_input)
            print(f"助手: {response}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
