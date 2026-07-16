import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import queue
import threading
import torch

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', 
                              model='silero_vad', 
                              force_reload=False
                              )

fs = 16000

audio_queue = queue.Queue(maxsize=50)
buffer = []

def callback(indata, frames, time, status):
    if status:
        print(status)
    data = indata.copy()
    audio_tensor = torch.from_numpy(data.flatten()).float()
    speech_prob = model(audio_tensor, fs).item()
    if speech_prob > 0.5:
        print("speech")
        audio_queue.put(indata.copy())
    else:
        print("silence")

def consumer():
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        buffer.append(chunk)

consumer_thread = threading.Thread(target=consumer)
consumer_thread.start()

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
    tup = tuple(buffer)
    fin = np.concatenate(tup)
    write("scratch/live_test.wav", fs, fin)
print("done")
