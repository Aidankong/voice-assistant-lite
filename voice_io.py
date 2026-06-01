#!/usr/bin/env python3
"""
语音输入/输出模块
支持麦克风录音、语音识别、语音合成和播放
"""

import asyncio
import numpy as np
import wave
from typing import Optional
from faster_whisper import WhisperModel
import edge_tts
import subprocess


class VoiceInput:
    """语音输入 - 录音 + 语音识别"""

    def __init__(self, model_size: str = "tiny", device: str = "cuda"):
        """
        初始化语音输入

        Args:
            model_size: Whisper 模型大小 (tiny/base/medium/large)
            device: 运行设备 (cuda/cpu)
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self._init_model()

    def _init_model(self):
        """初始化 Whisper 模型"""
        try:
            print(f"[VoiceInput] 加载 Whisper 模型 ({self.model_size})...")
            # 优先使用 ModelScope 下载的模型
            model_path = "~/.cache/modelscope/Systran/faster-whisper-tiny"

            # 检查 ModelScope 模型是否存在
            import os
            expanded_path = os.path.expanduser(model_path)
            if os.path.exists(expanded_path):
                print(f"[VoiceInput] 使用 ModelScope 模型: {expanded_path}")
                self.model = WhisperModel(
                    expanded_path,
                    device=self.device,
                    compute_type="float16" if self.device == "cuda" else "int8"
                )
            else:
                # 使用 HuggingFace 模型
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="float16" if self.device == "cuda" else "int8"
                )
            print(f"[VoiceInput] Whisper 模型加载完成")
        except Exception as e:
            print(f"[VoiceInput] Whisper 模型加载失败: {e}")

    def record_audio(self, duration: int = 5, sample_rate: int = 16000) -> bytes:
        """
        录音

        Args:
            duration: 录音时长（秒）
            sample_rate: 采样率

        Returns:
            音频数据 (bytes)
        """
        print(f"[VoiceInput] 录音 {duration} 秒...")

        try:
            # 使用 arecord 录音 (Linux)
            cmd = [
                "arecord",
                "-q",  # 安静模式
                "-t", "wav",  # 输出格式
                "-f", "S16_LE",  # 格式
                "-c", "1",  # 单声道
                "-r", str(sample_rate),  # 采样率
                "-d", "default",  # 设备
                "-"  # 输出到 stdout
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # 录音指定时长
            audio_data, _ = process.communicate(timeout=duration)
            return audio_data

        except subprocess.TimeoutExpired:
            process.kill()
            return b""
        except Exception as e:
            print(f"[VoiceInput] 录音失败: {e}")
            # 尝试使用 pyaudio
            return self._record_with_pyaudio(duration, sample_rate)

    def _record_with_pyaudio(self, duration: int, sample_rate: int) -> bytes:
        """使用 pyaudio 录音"""
        try:
            import pyaudio

            p = pyaudio.PyAudio()

            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=1024
            )

            frames = []
            for _ in range(int(sample_rate / 1024 * duration)):
                frames.append(stream.read(1024))

            stream.stop_stream()
            stream.close()
            p.terminate()

            return b"".join(frames)

        except ImportError:
            print("[VoiceInput] pyaudio 未安装，无法录音")
            return b""
        except Exception as e:
            print(f"[VoiceInput] pyaudio 录音失败: {e}")
            return b""

    def transcribe(self, audio_data: bytes, language: str = "zh") -> Optional[str]:
        """
        语音识别

        Args:
            audio_data: 音频数据
            language: 语言代码

        Returns:
            识别的文本
        """
        if not self.model or not audio_data:
            return None

        try:
            print("[VoiceInput] 正在识别...")
            segments, info = self.model.transcribe(
                audio_data,
                language=language,
                beam_size=5
            )

            text = "".join([seg.text for seg in segments])
            print(f"[VoiceInput] 识别结果: {text}")
            return text.strip()

        except Exception as e:
            print(f"[VoiceInput] 识别失败: {e}")
            return None


class VoiceOutput:
    """语音输出 - 语音合成 + 播放"""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        """
        初始化语音输出

        Args:
            voice: TTS 语音
        """
        self.voice = voice
        print(f"[VoiceOutput] TTS 语音: {self.voice}")

    async def text_to_speech(self, text: str) -> bytes:
        """
        文本转语音

        Args:
            text: 输入文本

        Returns:
            音频数据 (bytes)
        """
        if not text:
            return b""

        try:
            communicate = edge_tts.Communicate(text, self.voice)

            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            return audio_data

        except Exception as e:
            print(f"[VoiceOutput] TTS 失败: {e}")
            return b""

    def play_audio(self, audio_data: bytes):
        """
        播放音频

        Args:
            audio_data: 音频数据
        """
        if not audio_data:
            return

        try:
            # 使用 aplay 播放 (Linux)
            subprocess.run(
                ["aplay", "-q"],
                input=audio_data,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            print(f"[VoiceOutput] 播放失败: {e}")

    async def speak(self, text: str):
        """
        文本转语音并播放

        Args:
            text: 输入文本
        """
        print(f"[VoiceOutput] 语音: {text}")
        audio_data = await self.text_to_speech(text)
        if audio_data:
            self.play_audio(audio_data)


# ==================== 测试 ====================

async def test_voice_io():
    """测试语音输入输出"""

    print("="*50)
    print("语音 I/O 测试")
    print("="*50)

    # 测试 TTS
    print("\n测试 TTS...")
    voice_out = VoiceOutput()
    await voice_out.speak("你好，我是 Jarvis")

    # 测试 ASR (需要麦克风)
    print("\n测试 ASR (按 Ctrl+C 跳过)...")
    try:
        voice_in = VoiceInput()

        print("请说话...")
        audio = voice_in.record_audio(duration=3)

        if audio:
            text = voice_in.transcribe(audio)
            if text:
                print(f"你说: {text}")

                # 重复一遍
                await voice_out.speak(f"你说的是: {text}")

    except KeyboardInterrupt:
        print("\n测试已跳过")


if __name__ == "__main__":
    asyncio.run(test_voice_io())
