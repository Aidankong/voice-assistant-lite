#!/usr/bin/env python3
import os
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

from voice_io import VoiceInput

print('=== 测试语音识别 ===')
vi = VoiceInput('tiny', 'cpu')
print('录音 3 秒...')
audio = vi.record_audio(duration=3)
if audio:
    print(f'录音完成: {len(audio)} bytes')
    print('识别中...')
    text = vi.transcribe(audio)
    print(f'识别结果: {text}')
else:
    print('录音失败')
