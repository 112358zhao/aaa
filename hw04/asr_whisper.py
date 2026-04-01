import whisper
import time
import sys

def transcribe_audio_file(model, audio_path):
    """识别音频文件"""
    print(f"正在识别音频文件: {audio_path}")
    start = time.time()
    result = model.transcribe(audio_path, language="zh", fp16=False)
    elapsed = time.time() - start
    print(f"识别完成，耗时: {elapsed:.2f} 秒")
    print("识别结果:")
    print(result["text"])
    return result["text"]

def transcribe_microphone(model, duration=5, sample_rate=16000):
    """实时麦克风输入（需sounddevice）"""
    import sounddevice as sd
    import numpy as np
    import tempfile
    import wave

    print(f"正在录制 {duration} 秒麦克风输入...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait()
    
    # 保存为临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((recording * 32767).astype(np.int16).tobytes())
    
    print("正在识别...")
    result = model.transcribe(temp_path, language="zh", fp16=False)
    print("识别结果:", result["text"])
    return result["text"]

if __name__ == "__main__":
    # 加载模型（tiny/base/small/medium/large）
    model = whisper.load_model("tiny")  # tiny模型约39MB，CPU可运行
    print("模型加载完成")

    # 模式选择
    if len(sys.argv) > 1 and sys.argv[1] == "mic":
        transcribe_microphone(model)
    else:
        # 默认测试音频文件（请替换为你的音频路径）
        test_audio = 'C:/Users/19277/cloned_voice.mp3'  # 或任务二导出的 cloned_voice.mp3
        transcribe_audio_file(model, test_audio)