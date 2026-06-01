# 方案 A：LangGraph 语音助手架构

> **适用场景**：需要复杂状态管理、未来扩展性强、社区支持好的项目

## 一、系统总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           高性能电脑端（主控）                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        LangGraph 工作流                               │  │
│   │                                                                       │  │
│   │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │  │
│   │   │   ASR    │───▶│ Intent   │───▶│  Tool    │───▶│ Response │      │  │
│   │   │  Node    │    │ Classify │    │ Execute  │    │ Generate │      │  │
│   │   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘      │  │
│   │        │               │               │               │            │  │
│   │        ▼               ▼               ▼               ▼            │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │                     Shared State                            │   │  │
│   │   │  • transcript (用户输入)                                     │   │  │
│   │   │  • intent (意图分类)                                         │   │  │
│   │   │  • tools (需要调用的工具)                                    │   │  │
│   │   │  • context (对话上下文)                                      │   │  │
│   │   │  • memory (长期记忆)                                         │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                                                                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    │ ZeroMQ / gRPC                          │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        香橙派端（执行层）                              │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │  │
│   │   │ ROS2 Bridge  │  │ HA Controller │  │  Future: 机械臂/底盘等   │  │  │
│   │   │  节点        │  │   节点        │  │           控制节点          │  │  │
│   │   └──────────────┘  └──────────────┘  └──────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 二、技术栈选型

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| **LLM** | Qwen2.5-7B-Instruct-Q4_K_M | 速度快、中文强、工具调用原生支持 |
| **Agent框架** | LangGraph | 状态机模型、生产级、社区大 |
| **ASR** | faster-whisper (tiny/base) | 实时语音识别 |
| **TTS** | edge-tts / Piper | 语音合成 |
| **记忆系统** | PostgreSQL + pgvector | 向量存储 + 结构化记忆 |
| **可观测性** | LangSmith | 调试、追踪、评估 |
| **通信协议** | ZeroMQ | 低延迟、可靠 |

## 三、LangGraph 工作流设计

### 3.1 状态定义

```python
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import add_messages
import operator

class VoiceAssistantState(TypedDict):
    """语音助手状态定义"""

    # 对话消息（LangGraph 内置）
    messages: Annotated[list, add_messages]

    # 用户输入
    transcript: str                      # 语音识别文本

    # 意图理解
    intent: str                          # 意图类别：light_on/light_off/arm_move/chat等
    confidence: float                    # 意图置信度
    entities: dict                       # 提取的实体：{"room": "卧室", "color": "暖色"}

    # 工具执行
    tools_needed: List[str]              # 需要调用的工具列表
    tool_results: List[dict]             # 工具执行结果

    # 记忆
    context_summary: str                 # 当前对话上下文摘要
    retrieved_memories: List[dict]       # 从长期记忆检索的相关信息

    # 响应
    response_text: str                   # 生成的回复文本
    response_audio: bytes                # TTS 生成的音频

    # 元数据
    timestamp: float                     # 时间戳
    session_id: str                      # 会话ID
```

### 3.2 节点定义

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

# ==================== 节点函数 ====================

async def asr_node(state: VoiceAssistantState) -> VoiceAssistantState:
    """语音识别节点"""
    # 接收唤醒词触发后的音频流
    audio = state.get("audio_input")

    # 调用 faster-whisper
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio, language="zh")

    transcript = "".join([seg.text for seg in segments])

    return {
        "transcript": transcript,
        "timestamp": time.time()
    }


async def intent_classifier_node(state: VoiceAssistantState) -> VoiceAssistantState:
    """意图分类节点 - 使用 LLM 进行意图识别和实体提取"""

    prompt = f"""分析用户语音指令，提取意图和实体。

用户输入：{state['transcript']}

请以 JSON 格式返回：
{{
    "intent": "意图类型 (light_on/light_off/light_color/arm_move/arm_pick/navigation/chat/unknown)",
    "confidence": 0.0-1.0,
    "entities": {{"entity_key": "value"}},
    "needs_tools": ["tool1", "tool2"]
}}

可用工具：
- ha_control: 控制HomeAssistant设备（灯光、开关等）
- arm_control: 控制机械臂（移动、抓取）
- base_control: 控制移动底盘（导航、移动）
- sensor_query: 查询传感器状态
"""

    from ollama import Client
    client = Client(host='localhost:11434')
    response = client.generate(model='qwen2.5:7b', prompt=prompt, format='json')
    result = json.loads(response['response'])

    return {
        "intent": result['intent'],
        "confidence": result['confidence'],
        "entities": result['entities'],
        "tools_needed": result['needs_tools']
    }


async def memory_retrieval_node(state: VoiceAssistantState) -> VoiceAssistantState:
    """记忆检索节点 - 从向量数据库检索相关记忆"""

    # 使用当前对话作为查询向量
    query_text = state['transcript']

    # 向量检索
    from pgvector psycopg
    # ... pgvector 查询逻辑

    # 或者使用轻量级方案 ChromaDB
    import chromadb
    chroma_client = chromadb.Client()
    collection = chroma_client.get_collection("voice_memories")
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )

    return {
        "retrieved_memories": results['documents'][0]
    }


async def tool_execute_node(state: VoiceAssistantState) -> VoiceAssistantState:
    """工具执行节点 - 通过 ROS2 桥接执行实际操作"""

    results = []

    for tool_name in state['tools_needed']:
        # 通过 ZeroMQ 发送到香橙派
        import zmq
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://orangepi:5555")

        request = {
            "tool": tool_name,
            "params": state['entities']
        }
        socket.send_json(request)

        # 等待结果
        result = socket.recv_json()
        results.append(result)
        socket.close()

    return {
        "tool_results": results
    }


async def response_generate_node(state: VoiceAssistantState) -> VoiceAssistantState:
    """回复生成节点 - 生成自然语言回复"""

    # 构建上下文
    context_parts = [
        f"用户输入：{state['transcript']}",
        f"识别意图：{state['intent']}",
    ]

    if state.get('retrieved_memories'):
        context_parts.append(f"相关记忆：{state['retrieved_memories']}")

    if state.get('tool_results'):
        context_parts.append(f"执行结果：{state['tool_results']}")

    prompt = f"""你是智能家居语音助手。根据以下信息生成简洁、自然的回复。

{chr(10).join(context_parts)}

要求：
1. 回复要简洁，控制在20字以内
2. 语气自然、友好
3. 确认执行的操作
4. 如果是闲聊，自然对话
"""

    from ollama import Client
    client = Client(host='localhost:11434')
    response = client.generate(model='qwen2.5:7b', prompt=prompt)

    return {
        "response_text": response['response']
    }


async def tts_node(state: VoiceAssistantState) -> VoiceAssistantState:
    """语音合成节点"""

    import edge_tts

    text = state['response_text']
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")

    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    # 播放音频
    # ... 发送到香橙派或本地播放

    return {
        "response_audio": audio_data
    }


async def memory_store_node(state: VoiceAssistantState) -> VoiceAssistantState:
    """记忆存储节点 - 存储重要交互到长期记忆"""

    # 判断是否需要存储
    if state['intent'] != 'chat':  # 只存储有意义的操作，不存闲聊
        memory_entry = {
            "timestamp": state['timestamp'],
            "intent": state['intent'],
            "entities": state['entities'],
            "transcript": state['transcript'],
            "result": state.get('tool_results', [])
        }

        # 存储到向量数据库
        # ... 存储逻辑

    return state
```

### 3.3 工作流图构建

```python
from langgraph.graph import StateGraph, END

def create_voice_assistant_graph():
    """构建语音助手工作流图"""

    workflow = StateGraph(VoiceAssistantState)

    # 添加节点
    workflow.add_node("asr", asr_node)
    workflow.add_node("intent_classifier", intent_classifier_node)
    workflow.add_node("memory_retrieve", memory_retrieval_node)
    workflow.add_node("tool_execute", tool_execute_node)
    workflow.add_node("response_generate", response_generate_node)
    workflow.add_node("tts", tts_node)
    workflow.add_node("memory_store", memory_store_node)

    # 设置入口
    workflow.set_entry_point("asr")

    # 添加边（固定流转）
    workflow.add_edge("asr", "intent_classifier")
    workflow.add_edge("intent_classifier", "memory_retrieve")

    # 条件边：根据是否需要工具决定路径
    def should_use_tools(state: VoiceAssistantState) -> str:
        return "tool_execute" if state.get('tools_needed') else "response_generate"

    workflow.add_conditional_edges(
        "memory_retrieve",
        should_use_tools,
        {
            "tool_execute": "tool_execute",
            "response_generate": "response_generate"
        }
    )

    workflow.add_edge("tool_execute", "response_generate")
    workflow.add_edge("response_generate", "tts")
    workflow.add_edge("tts", "memory_store")
    workflow.add_edge("memory_store", END)

    # 添加持久化（PostgreSQL）
    checkpointer = PostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost/voice_assistant"
    )

    return workflow.compile(checkpointer=checkpointer)
```

## 四、ROS2 桥接设计

### 4.1 香橙派端 ROS2 节点结构

```python
# hermes_bridge.py - 香橙派运行

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import zmq
import json

class HermesBridgeNode(Node):
    """Hermes Agent → ROS2 桥接节点"""

    def __init__(self):
        super().__init__('hermes_bridge_node')

        # ZeroMQ 服务器（接收来自高性能电脑的指令）
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")

        # ROS2 发布者
        self.light_cmd_pub = self.create_publisher(String, 'home/light/command', 10)
        self.arm_cmd_pub = self.create_publisher(String, 'arm/command', 10)
        self.base_cmd_pub = self.create_publisher(String, 'base/command', 10)

        # 启动接收线程
        import threading
        threading.Thread(target=self._receive_commands, daemon=True).start()

    def _receive_commands(self):
        """持续接收来自 LangGraph 的指令"""
        while rclpy.ok():
            # 接收指令
            request = self.socket.recv_json()
            self.get_logger().info(f"收到指令: {request}")

            # 根据工具类型分发到对应的 ROS2 话题
            tool = request.get('tool')
            params = request.get('params', {})

            result = self._execute_tool(tool, params)

            # 返回结果
            self.socket.send_json(result)

    def _execute_tool(self, tool: str, params: dict) -> dict:
        """执行工具指令"""

        if tool == "ha_control":
            return self._ha_control(params)
        elif tool == "arm_control":
            return self._arm_control(params)
        elif tool == "base_control":
            return self._base_control(params)
        elif tool == "sensor_query":
            return self._sensor_query(params)
        else:
            return {"success": False, "error": "Unknown tool"}

    def _ha_control(self, params: dict) -> dict:
        """Home Assistant 设备控制"""
        # 发布到 ROS2 话题，由另一个节点处理 HA 通信
        msg = String()
        msg.data = json.dumps(params)
        self.light_cmd_pub.publish(msg)
        return {"success": True, "message": "HA command sent"}

    def _arm_control(self, params: dict) -> dict:
        """机械臂控制"""
        msg = String()
        msg.data = json.dumps(params)
        self.arm_cmd_pub.publish(msg)
        return {"success": True, "message": "Arm command sent"}

    def _base_control(self, params: dict) -> dict:
        """移动底盘控制"""
        msg = String()
        msg.data = json.dumps(params)
        self.base_cmd_pub.publish(msg)
        return {"success": True, "message": "Base command sent"}

    def _sensor_query(self, params: dict) -> dict:
        """传感器状态查询"""
        # 实现传感器查询逻辑
        return {"success": True, "data": {}}


def main():
    rclpy.init()
    node = HermesBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 4.2 HA 控制节点

```python
# ha_controller.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import json

class HAControllerNode(Node):
    """Home Assistant 控制节点"""

    def __init__(self):
        super().__init__('ha_controller_node')

        # HA 配置
        self.ha_url = "http://homeassistant.local:8123"
        self.ha_token = "YOUR_LONG_LIVED_ACCESS_TOKEN"

        # 订阅来自 hermes_bridge 的指令
        self.subscription = self.create_subscription(
            String,
            'home/light/command',
            self.handle_command,
            10
        )

    def handle_command(self, msg: String):
        """处理 HA 设备控制指令"""
        command = json.loads(msg.data)

        entity_id = command.get('entity_id')
        action = command.get('action')  # on/off/toggle

        self.get_logger().info(f"控制 HA 设备: {entity_id} -> {action}")

        # 调用 HA API
        if action == "on":
            self._call_service("homeassistant/turn_on", entity_id)
        elif action == "off":
            self._call_service("homeassistant/turn_off", entity_id)

    def _call_service(self, service: str, entity_id: str, **kwargs):
        """调用 Home Assistant 服务"""
        url = f"{self.ha_url}/api/services/{service}"
        headers = {
            "Authorization": f"Bearer {self.ha_token}",
            "content-type": "application/json"
        }
        payload = {
            "entity_id": entity_id,
            **kwargs
        }
        response = requests.post(url, json=payload, headers=headers)
        return response.json()


def main():
    rclpy.init()
    node = HAControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 五、部署架构

### 5.1 高性能电脑服务

```
┌─────────────────────────────────────────────────────────────┐
│                    高性能电脑服务                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Ollama/LM Studio│    │  faster-whisper│                │
│  │  (本地 LLM 服务)  │    │  (ASR 服务)     │                │
│  └────────┬─────────┘    └────────┬─────────┘                │
│           │                      │                           │
│           └──────────┬───────────┘                           │
│                      ▼                                       │
│           ┌─────────────────────────┐                       │
│           │    LangGraph 服务        │                       │
│           │  (FastAPI / gRPC)        │                       │
│           └───────────┬─────────────┘                       │
│                       │                                       │
│                       ▼                                       │
│           ┌─────────────────────────┐                       │
│           │    PostgreSQL + pgvector │                       │
│           │    (记忆持久化)           │                       │
│           └─────────────────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Docker Compose 配置

```yaml
# docker-compose.yml

version: '3.8'

services:
  # PostgreSQL + pgvector
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: voice_assistant
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Redis（可选，用于缓存）
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  # LangGraph 服务
  langgraph-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_URL=postgresql://postgres:postgres@postgres/voice_assistant
      - OLLAMA_HOST=http://ollama:11434
      - ZMQ_ORANGE_PI=orangepi:5555
    depends_on:
      - postgres
      - ollama

  # Ollama
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

volumes:
  postgres_data:
  ollama_data:
```

## 六、预期性能

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **唤醒词检测** | <50ms | openWakeWord，可跑在香橙派 |
| **语音识别** | 200-400ms | faster-whisper base/tiny |
| **意图分类** | 100-200ms | Qwen2.5-7B-Q4，简单指令 |
| **工具执行** | 50-200ms | 取决于设备响应速度 |
| **回复生成** | 100-300ms | Qwen2.5-7B-Q4 |
| **语音合成** | 100-200ms | edge-tts |
| **端到端延迟** | <1.5s | 从说话结束到听到回复 |

## 七、优缺点分析

### 优点

| 方面 | 说明 |
|------|------|
| **成熟稳定** | LangChain 团队维护，生产级验证 |
| **状态管理** | 内置状态机，对话流清晰可控 |
| **持久化** | Checkpointing 自动保存对话状态 |
| **可观测性** | LangSmith 提供完整的调试和追踪 |
| **社区支持** | 最大的社区，问题容易解决 |
| **扩展性** | 易于添加新节点、新工具 |

### 缺点

| 方面 | 说明 |
|------|------|
| **学习曲线** | 需要理解 LangGraph 的概念 |
| **框架重量** | 对于简单场景可能过重 |
| **依赖较多** | LangChain 生态依赖 |

## 八、启动步骤

```bash
# 1. 安装依赖
pip install langgraph langchain-langchain postgresql-client

# 2. 启动数据库
docker-compose up -d postgres redis

# 3. 启动 Ollama 并拉取模型
ollama serve
ollama pull qwen2.5:7b

# 4. 运行 LangGraph 服务
python -m voice_assistant.main

# 5. 在香橙派启动 ROS2 节点
python3 hermes_bridge.py
python3 ha_controller.py
```

## 九、文件结构

```
voice-assistant-langgraph/
├── docker-compose.yml
├── requirements.txt
├── .env
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 服务入口
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── workflow.py         # LangGraph 工作流定义
│   │   ├── nodes/
│   │   │   ├── asr.py
│   │   │   ├── intent.py
│   │   │   ├── tools.py
│   │   │   ├── response.py
│   │   │   └── tts.py
│   │   └── state.py            # 状态定义
│   ├── tools/
│   │   ├── ros2_bridge.py      # ROS2 通信
│   │   └── memory.py           # 记忆管理
│   └── config.py
└── README.md
```

---

**文档版本**: v1.0
**更新日期**: 2026-06-01
**作者**: Claude
