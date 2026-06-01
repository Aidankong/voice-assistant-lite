#!/usr/bin/env python3
"""
唤醒词检测模块
支持 "Jarvis" 唤醒词检测
"""

import asyncio
import time
from typing import Callable, Optional
import numpy as np


class WakeWordDetector:
    """唤醒词检测器"""

    def __init__(self, wake_word: str = "Jarvis"):
        self.wake_word = wake_word.lower()
        self.is_running = False
        self.on_wake_callback: Optional[Callable] = None

        # 检测模式
        self.mode = "asr"  # asr (实时语音识别) 或 model (专用模型)

    def set_callback(self, callback: Callable):
        """设置唤醒回调函数"""
        self.on_wake_callback = callback

    def check_text(self, text: str) -> bool:
        """
        检查文本中是否包含唤醒词

        Args:
            text: 要检查的文本

        Returns:
            bool: 是否检测到唤醒词
        """
        if not text:
            return False

        text_lower = text.lower()

        # 精确匹配
        if self.wake_word in text_lower:
            return True

        # 模糊匹配（语音识别可能的变体）
        variations = [
            self.wake_word,
            "jarvis",
            "ja vis",
            "java is",
            "jahvis",
        ]
        for variation in variations:
            if variation in text_lower.replace(" ", ""):
                return True

        return False

    async def start_asr_mode(self, asr_function: Callable):
        """
        使用 ASR 模式检测唤醒词

        Args:
            asr_function: 语音识别函数，返回识别的文本
        """
        self.is_running = True
        print(f"[唤醒词检测] 启动 ASR 模式，监听唤醒词: {self.wake_word}")

        while self.is_running:
            try:
                # 调用 ASR 获取文本
                text = await asr_function()

                if text and self.check_text(text):
                    print(f"[唤醒词检测] 检测到唤醒词: {text}")
                    if self.on_wake_callback:
                        await self.on_wake_callback()

                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"[唤醒词检测] 错误: {e}")
                await asyncio.sleep(1)

    def start_model_mode(self):
        """
        使用专用模型模式检测唤醒词
        需要音频流输入
        """
        # TODO: 实现 openWakeWord 或 Porcupine 集成
        print("[唤醒词检测] 模型模式待实现")
        pass

    def stop(self):
        """停止检测"""
        self.is_running = False
        print("[唤醒词检测] 已停止")


# ==================== 测试 ====================

async def test_wake_word():
    """测试唤醒词检测"""

    detector = WakeWordDetector("Jarvis")

    # 测试用例
    test_cases = [
        "Jarvis",
        "嘿 Jarvis",
        "jarvis 帮我开灯",
        "贾维斯",
        "你好助手",
        "",
        "Hello Jarvis",
    ]

    print("="*50)
    print("唤醒词检测测试")
    print("="*50)

    for test in test_cases:
        result = detector.check_text(test)
        status = "✓" if result else "✗"
        print(f"{status} '{test}' -> {result}")

    print("\n" + "="*50)
    print("测试完成")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(test_wake_word())
