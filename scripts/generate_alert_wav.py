"""
Generate a short alert WAV file (1kHz sine, 0.3s) at `assets/alert.wav`.
Run: python scripts/generate_alert_wav.py
"""
import math
import wave
import struct
from pathlib import Path

OUT = Path("static")
OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / "alert.wav"

framerate = 44100
duration = 0.3
frequency = 1000.0
amplitude = 16000

nframes = int(framerate * duration)
with wave.open(str(OUT_FILE), 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(framerate)
    for i in range(nframes):
        t = float(i) / framerate
        val = int(amplitude * math.sin(2.0 * math.pi * frequency * t))
        data = struct.pack('<h', val)
        wf.writeframesraw(data)

print(f"Wrote: {OUT_FILE}")
