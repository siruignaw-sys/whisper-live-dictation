### **whisper-live-dictation**
Live, pause-to-commit dictation using local Whisper transcription. Talk continuously, and the tool will detect natural pauses, transcribing each sentence in the background, and printing it, without waiting for the whole session to end. 

Built to use Whisper `large-v3` for high accuracy (including mixed Chinese/English speech and technicaly vocabulary), while still feeling responsive, by transcribing sentence-by-sentence instead of all at once. 

## How it works
 - Captures audio continuously from the microphone
 - Runs Silero VAD to detect speech vs. silence in real time
 - A segmenter watches for ~700ms of silence to mark the end of a sentence (with a small buffer so the start of speech isn't clipped)
 - Completed segments are queued and transcribed in the background by `faster-whisper` (Whisper `large-v3`, GPU-accelerated)
 - Transcribed text is printed as each segment completes

 ## Reguirements
 * Python 3.10+
 * NVIDIA GPU with CUDA support: `large-v3` needs a real GPU to run at usable speed; 8GB+ VRAM reccomended
 * A working microphone

 ## Setup
 ```bash
 git clone https://github.com/siruignaw-sys/whisper-live-dictation
 cd whisper-live-dictation
 python3 -m venv whisper-live-env
 source whisper-live-env/bin/activate   # or whisper-live-env\Scripts\activate on Windows
 pip install -r requirements.txt
 ```

 ## CUDA/cuDNN note
 `faster-whisper` needs CUDA runtime libraries (cuBLAS, cuDNN) available, separate from having an NVIDIA driver installed. If you hit an error like `Library cublas64_12.dll is not found` (Windows) or an equivalent `.so` loading error (Linux), you likely need to either:
 * Install the matching CUDA Toolkit (12.x) directly from NVIDIA, or
 * Ensure th `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` pip packages (already in `requirements.txt`) are actually discoverable on your system's library path

 This was the most finnicky part of getting this running. If you hit it, don't assume something's fundamentally broken, it usually just needs the library path pointed in the right place.

 ## Running it
 ```bash
 python src/mic_capture.py
 ```
 Talk normally, pause between sentences; each sentence should print shortly after you finish saying it. Press `Ctrl+C` to stop.

 ## Known limitations (current state)
 * **Occasional language mix-ups.** Very rarely, a segment may transcribe in the wrong language even with the language explicity set. Uncommon, not yet understood.
 * **Output is console-only right now.** No GUI/live text window yet, text just prints to the terminal.
 * **Punctuation and fluency are noticeably better in English than Chinese.** This is a general Whisper model limitation, not specific to this tool.
 * **Technical/uncommon terms can occasionally be mis-transcribed** as a phonetically similar but wrong word (e.g. "site-directed" -> "cytodirected"). Worth double-checking output on unfamiliar or high technical vocabulary. 

 ## Project status
 Early working prototype. Core pipeline (capture -> VAD -> segmentation -> transcription) is functional and tested against real mixed Chinese/English speech, including technical vocabulary. Not yet packaged for non-technical use; currently requires running from source with Python installed.