#!/usr/bin/env python3
import wave
import numpy as np

with wave.open('/tmp/test_voice.wav', 'rb') as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / rate
    print(f'采样率: {rate} Hz, 帧数: {frames}, 时长: {duration:.2f} 秒')

    audio_data = f.readframes(frames)
    samples = np.frombuffer(audio_data, dtype=np.int16)

    print(f'样本数: {len(samples)}')
    print(f'最大值: {samples.max()}, 最小值: {samples.min()}')
    print(f'平均值: {samples.mean():.2f}')
    print(f'标准差: {samples.std():.2f}')

    loud_samples = np.abs(samples) > 1000
    pct = loud_samples.sum() / len(samples) * 100
    print(f'大于阈值的样本占比: {pct:.2f}%')

    if pct < 0.01:
        print('音频基本为静音')
    else:
        print('音频包含声音')
