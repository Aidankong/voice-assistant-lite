#!/usr/bin/env python3
"""
语音输入/输出模块
支持麦克风录音、语音识别、语音合成和播放
"""

import asyncio
import numpy as np
import wave
import os
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
            # ModelScope 模型路径列表
            import os
            possible_paths = [
                "~/.cache/whisper/~/.cache/modelscope/Systran/faster-whisper-tiny",
                "~/.cache/modelscope/Systran/faster-whisper-tiny",
                "~/.cache/huggingface/hub/models--Systran--faster-whisper-tiny",
            ]

            model_loaded = False
            for path in possible_paths:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    print(f"[VoiceInput] 使用本地模型: {expanded_path}")
                    self.model = WhisperModel(
                        expanded_path,
                        device=self.device,
                        compute_type="float16" if self.device == "cuda" else "int8"
                    )
                    model_loaded = True
                    break

            if not model_loaded:
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
            # 使用 arecord 录音 (Linux) - 直接输出到文件
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            cmd = [
                "arecord",
                "-q",  # 安静模式
                "-d", "3",  # 录音时长
                "-f", "S16_LE",  # 格式
                "-c", "1",  # 单声道
                "-r", str(sample_rate),  # 采样率
                temp_path
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=duration + 5)

            if result.returncode == 0:
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()
                os.unlink(temp_path)
                return audio_data
            else:
                os.unlink(temp_path)
                print(f"[VoiceInput] arecord 错误: {result.stderr.decode()}")
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
            # 将音频数据保存到临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(audio_data)

            # 使用文件路径进行识别
            segments, info = self.model.transcribe(
                temp_path,
                language=language,
                beam_size=5
            )

            # 删除临时文件
            os.unlink(temp_path)

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
