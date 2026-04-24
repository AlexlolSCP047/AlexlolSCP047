#!/usr/bin/env python3
"""Transcribe audio with word-level timestamps and write SRT/JSON."""
import json
import sys
from faster_whisper import WhisperModel

AUDIO = "audio.wav"
SRT_OUT = "subs.srt"
JSON_OUT = "words.json"

print("Loading Whisper model (small)...", flush=True)
model = WhisperModel("small", device="cpu", compute_type="int8")

print("Transcribing...", flush=True)
segments, info = model.transcribe(
    AUDIO,
    word_timestamps=True,
    vad_filter=True,
    beam_size=5,
)

print(f"Detected language: {info.language} (prob {info.language_probability:.2f})", flush=True)

def fmt_ts(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

srt_lines = []
words_data = []
seg_data = []
idx = 1

for seg in segments:
    seg_words = []
    if seg.words:
        for w in seg.words:
            words_data.append({
                "word": w.word,
                "start": float(w.start) if w.start is not None else float(seg.start),
                "end": float(w.end) if w.end is not None else float(seg.end),
            })
            seg_words.append({
                "word": w.word,
                "start": float(w.start) if w.start is not None else float(seg.start),
                "end": float(w.end) if w.end is not None else float(seg.end),
            })
    seg_data.append({
        "start": float(seg.start),
        "end": float(seg.end),
        "text": seg.text.strip(),
        "words": seg_words,
    })
    text = seg.text.strip()
    srt_lines.append(str(idx))
    srt_lines.append(f"{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}")
    srt_lines.append(text)
    srt_lines.append("")
    idx += 1
    print(f"[{seg.start:6.2f}-{seg.end:6.2f}] {text}", flush=True)

with open(SRT_OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(srt_lines))

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump({
        "language": info.language,
        "duration": info.duration,
        "segments": seg_data,
        "words": words_data,
    }, f, ensure_ascii=False, indent=2)

print(f"Wrote {SRT_OUT} and {JSON_OUT}", flush=True)
