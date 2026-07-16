import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import queue
import threading
import torch
import collections

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
    audio_queue.put(indata.copy())

def consumer():
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
                    segment_data = np.concatenate(tuple(segment_buffer))
                    write("scratch/segment_" + str(test_num) + ".wav", fs, segment_data)
                    segment_buffer = []
                    test_num+=1
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
print("done")
