#!/usr/bin/env python3
"""Transcribe with sherpa-onnx whisper-tiny in chunks for word-level-ish timestamps."""
import json
import wave
import sys
import sherpa_onnx

MODEL_DIR = "models/sherpa-onnx-whisper-small"
PREFIX = "small"
AUDIO = "audio.wav"

print("Building recognizer...", flush=True)
recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
    encoder=f"{MODEL_DIR}/{PREFIX}-encoder.onnx",
    decoder=f"{MODEL_DIR}/{PREFIX}-decoder.onnx",
    tokens=f"{MODEL_DIR}/{PREFIX}-tokens.txt",
    num_threads=4,
    decoding_method="greedy_search",
    debug=False,
    language="es",
    task="transcribe",
)

# Read full WAV
with wave.open(AUDIO, "rb") as wf:
    sr = wf.getframerate()
    nframes = wf.getnframes()
    raw = wf.readframes(nframes)

import numpy as np
audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
total_dur = len(audio) / sr
print(f"Audio: {total_dur:.2f}s @ {sr}Hz", flush=True)

# Whisper has a 30s limit. Chunk into 12s segments with overlap to reduce hallucination.
CHUNK = 12.0
OVERLAP = 0.5
segments = []
t = 0.0
while t < total_dur:
    start = t
    end = min(t + CHUNK, total_dur)
    a = int(start * sr)
    b = int(end * sr)
    chunk_audio = audio[a:b]
    if len(chunk_audio) < sr * 0.5:
        break
    s = recognizer.create_stream()
    s.accept_waveform(sr, chunk_audio)
    recognizer.decode_stream(s)
    text = s.result.text.strip()
    if text:
        segments.append({"start": start, "end": end, "text": text})
        print(f"[{start:6.2f}-{end:6.2f}] {text}", flush=True)
    t += CHUNK - OVERLAP

# Save
with open("segments.json", "w", encoding="utf-8") as f:
    json.dump({"segments": segments, "duration": total_dur}, f, ensure_ascii=False, indent=2)
print(f"Wrote {len(segments)} segments", flush=True)
