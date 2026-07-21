import os
import sys

if getattr(sys, 'frozen', False):
    nvidia_path = os.path.join(sys._MEIPASS, "nvidia")
    print(f"MEIPASS: {sys._MEIPASS}")
    print(f"Looking for cublas at: {os.path.join(nvidia_path, 'cublas', 'bin')}")
    print(f"Exists: {os.path.exists(os.path.join(nvidia_path, 'cublas', 'bin'))}")
    print(os.listdir(os.path.join(nvidia_path, "cublas", "bin")))
else:
    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    nvidia_path = os.path.join(venv_root, "Lib", "site-packages", "nvidia")

if sys.platform == "win32":
    os.add_dll_directory(os.path.join(nvidia_path, "cublas", "bin"))
    os.add_dll_directory(os.path.join(nvidia_path, "cudnn", "bin"))

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import queue
import threading
import torch
import collections
from faster_whisper import WhisperModel
import pyautogui

if sys.platform == "linux" or sys.platform == "linux2":
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type == "wayland":
        print("Error: Wayland detected. Auto-typing isn't supported on Wayland yet.")
        print("This tool requires X11 on linux.")
        sys.exit(1)


model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                              model='silero_vad', 
                              force_reload=False,
                              trust_repo=True
                              )


fs = 16000

audio_queue = queue.Queue(maxsize=50)
transcription_queue = queue.Queue(maxsize=50)
buffer = []

def type_text(text):
    pyautogui.write(text + " ")

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def consumer():
    seq_id = 0
    state = "waiting"
    pre_roll = collections.deque(maxlen=7)
    segment_buffer = []
    silent_chunk_count = 0
    SILENCE_THRESHOLD_CHUNKS = 22
    test_num = 1
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        data = chunk.flatten()
        audio_tensor = torch.from_numpy(data).float()
        speech_prob = model(audio_tensor, fs).item()
        is_speech = speech_prob > 0.5

        pre_roll.append(chunk)

        if state == "waiting":
            if not is_speech:
                pass
            else:
                state = "in_speech"
                pre_roll_lst = []
                while len(pre_roll) > 0:
                    pre_roll_lst.append(pre_roll.popleft())
                segment_buffer[len(segment_buffer):] = pre_roll_lst
        elif state == "in_speech":
            if is_speech:
                silent_chunk_count = 0
                segment_buffer.append(chunk)
            else:
                silent_chunk_count += 1
                if silent_chunk_count >= SILENCE_THRESHOLD_CHUNKS:
                    state = "waiting"
                    segment_data = np.concatenate(tuple(segment_buffer)).flatten()
                    transcription_queue.put((seq_id, segment_data))
                    seq_id += 1
                    # write("scratch/segment_" + str(test_num) + ".wav", fs, segment_data)
                    segment_buffer = []
                    test_num+=1
    if len(segment_buffer) != 0:
        segment_data = np.concatenate(tuple(segment_buffer)).flatten()
        transcription_queue.put((seq_id, segment_data))
        seq_id += 1
        # write("scratch/segment_" + str(test_num) + ".wav", fs, segment_data)
        segment_buffer = []
        test_num+=1

def transcription_worker():
    print("loading model")
    model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    print("done loading model")
    while True:
        item = transcription_queue.get()
        if item is None:
            break
        sequence_id, audio = item[0], item[1]
        segments, info = model.transcribe(audio, task = "transcribe")
        for seg in segments:
            if not seg.text.strip():
                continue
            type_text(seg.text.strip())
consumer_thread = threading.Thread(target=consumer)
consumer_thread.start()

transcription_thread = threading.Thread(target=transcription_worker)
transcription_thread.start()

stream = sd.InputStream(samplerate=fs, channels=1, callback=callback, blocksize=512)
stream.start()
print("start recording now, press ctrl+c to stop")
try:
    while True:
        sd.sleep(1000)
except KeyboardInterrupt:
    print("Stopping...")
    stream.stop()
    stream.close()
    audio_queue.put(None)
    consumer_thread.join()
    transcription_queue.put(None)
    transcription_thread.join()
print("done")
