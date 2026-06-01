"""
语音助手核心类 - 分布式架构
核心功能：意图识别 + 工具执行 + 记忆系统 + 语音 I/O
支持：本地音频处理、Orange Pi 分布式音频
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from ollama import Client
import os

# 清理代理环境变量
for var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(var, None)

from voice_io import VoiceInput, VoiceOutput
from orangepi_client import OrangePiClient

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
    """分布式语音助手核心类"""

    def __init__(self, config: dict = None):
        # 默认配置
        self.config = config or {}
        self.ollama_host = self.config.get("ollama_host", "localhost:11434")
        self.model_name = self.config.get("model", "qwen2.5:14b")
        self.orange_pi_addr = self.config.get("orange_pi_addr", "192.168.10.55:5556")
        self.wake_word = self.config.get("wake_word", "Jarvis")
        self.use_orange_pi_audio = self.config.get("use_orange_pi_audio", True)

        # 初始化 LLM 客户端
        self.llm = Client(host=self.ollama_host)

        # 初始化 Orange Pi 客户端
        self.orange_pi = None
        if self.use_orange_pi_audio:
            try:
                self.orange_pi = OrangePiClient(
                    host=self.orange_pi_addr.split(":")[0],
                    port=int(self.orange_pi_addr.split(":")[1])
                )
                print(f"[VoiceAssistant] Orange Pi 音频已启用")
            except Exception as e:
                print(f"[VoiceAssistant] Orange Pi 连接失败: {e}")
                self.use_orange_pi_audio = False

        # 初始化语音 I/O（备用或用于 TTS）
        self.voice_input = VoiceInput(model_size="tiny", device="cpu") if not self.use_orange_pi_audio else None
        self.voice_output = VoiceOutput()

        # 工具注册表
        self.tools: Dict[str, Callable] = {}

        # 对话上下文
        self.context: Optional[ConversationContext] = None
        self.waiting_for_command = False  # 是否等待指令

        print(f"[VoiceAssistant] 初始化完成，使用模型: {self.model_name}")
        print(f"[VoiceAssistant] 音频来源: {'Orange Pi' if self.use_orange_pi_audio else '本地'}")
        print(f"[VoiceAssistant] 唤醒词: {self.wake_word}")

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

    # ==================== 语音输入（Orange Pi） ====================

    async def listen_for_wake_word(self, duration: int = 3) -> Optional[str]:
        """
        监听唤醒词并获取命令

        Args:
            duration: 监听唤醒词的时长（秒）

        Returns:
            识别的命令文本，未检测到唤醒词返回 None
        """
        if not self.orange_pi:
            print("[VoiceAssistant] Orange Pi 未连接")
            return None

        print(f"[VoiceAssistant] 监听唤醒词 '{self.wake_word}' ({duration}秒)...")

        result = self.orange_pi.detect_wake_word(listen_duration=duration)

        if result and result.get("detected"):
            print("[VoiceAssistant] 检测到唤醒词！")
            audio_data = result.get("audio")

            if audio_data:
                # 使用本地 Whisper 识别
                if not self.voice_input:
                    self.voice_input = VoiceInput(model_size="tiny", device="cpu")

                text = self.voice_input.transcribe(audio_data)
                return text

        return None

    async def record_audio(self, duration: int = 5) -> Optional[bytes]:
        """
        录音

        Args:
            duration: 录音时长（秒）

        Returns:
            音频数据
        """
        if self.orange_pi:
            return self.orange_pi.record_audio(duration)
        elif self.voice_input:
            return self.voice_input.record_audio(duration)
        else:
            print("[VoiceAssistant] 无可用音频输入")
            return None

    # ==================== ROS2 工具 ====================

    async def light_on(self, params: Dict) -> Dict:
        """开灯"""
        if self.orange_pi:
            return self.orange_pi.light_on()
        return {"success": False, "error": "Orange Pi 未连接"}

    async def light_off(self, params: Dict) -> Dict:
        """关灯"""
        if self.orange_pi:
            return self.orange_pi.light_off()
        return {"success": False, "error": "Orange Pi 未连接"}

    async def arm_move(self, params: Dict) -> Dict:
        """机械臂移动"""
        if self.orange_pi:
            position = params.get("position", {})
            speed = params.get("speed", 50)
            return self.orange_pi.arm_move(position, speed)
        return {"success": False, "error": "Orange Pi 未连接"}

    async def sensor_query(self, params: Dict) -> Dict:
        """传感器查询"""
        if self.orange_pi:
            status = self.orange_pi.get_status()
            return {"success": True, "data": status}
        return {"success": False, "error": "Orange Pi 未连接"}

    async def capture_image(self, params: Dict) -> Dict:
        """采集图像"""
        if self.orange_pi:
            image = self.orange_pi.capture_image()
            if image:
                return {"success": True, "size": len(image)}
            return {"success": False, "error": "图像采集失败"}
        return {"success": False, "error": "Orange Pi 未连接"}


# ==================== 入口 ====================

async def main():
    """主函数 - 测试"""

    import sys

    # 检查命令行参数
    use_voice = "--voice" in sys.argv or "-v" in sys.argv

    # 创建助手实例
    config = {
        "ollama_host": "localhost:11434",
        "model": "qwen2.5:14b",
        "orange_pi_addr": "192.168.10.55:5556",
        "wake_word": "Jarvis",
        "use_orange_pi_audio": True
    }
    assistant = VoiceAssistant(config)

    # 注册 ROS2 工具
    assistant.register_tool("light_on", assistant.light_on, "打开灯光")
    assistant.register_tool("light_off", assistant.light_off, "关闭灯光")
    assistant.register_tool("arm_move", assistant.arm_move, "机械臂移动（参数：position=坐标）")
    assistant.register_tool("sensor_query", assistant.sensor_query, "查询传感器状态")
    assistant.register_tool("capture_image", assistant.capture_image, "采集图像")

    # 交互式测试
    mode = "语音" if use_voice else "文本"
    print("\n" + "="*50)
    print(f"语音助手测试模式（{mode}输入）")
    print(f"唤醒词: {assistant.wake_word}")
    print(f"音频来源: {'Orange Pi' if assistant.use_orange_pi_audio else '本地'}")
    print("输入 'quit' 退出，'voice' 切换到语音模式")
    print("="*50 + "\n")

    while True:
        try:
            if use_voice and assistant.use_orange_pi_audio:
                # 使用 Orange Pi 进行唤醒词检测和录音
                user_input = await assistant.listen_for_wake_word(duration=5)

                if user_input:
                    print(f"识别: {user_input}")
                else:
                    print("未检测到唤醒词，继续监听...")
                    continue
            elif use_voice:
                # 本地语音输入模式
                print("请说话...")
                audio_data = assistant.voice_input.record_audio(duration=3)

                if audio_data:
                    user_input = assistant.voice_input.transcribe(audio_data)
                    if user_input:
                        print(f"识别: {user_input}")
                    else:
                        print("未识别到语音")
                        continue
                else:
                    print("录音失败")
                    continue
            else:
                # 文本输入模式
                user_input = input("你: ")
                if user_input.lower() == 'voice':
                    use_voice = True
                    print("切换到语音模式")
                    continue

            if user_input.lower() in ['quit', 'exit', '退出']:
                break

            # 处理输入
            response = await assistant.process(user_input)
            print(f"助手: {response}")

            # 语音输出
            if use_voice:
                await assistant.voice_output.speak(response)

            print()  # 空行

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())
