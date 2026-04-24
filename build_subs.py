#!/usr/bin/env python3
"""Build a fancy animated ASS subtitle from segments.json with word-level karaoke timing."""
import json
import re

with open("segments.json", "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["segments"]
duration = data["duration"]

# Whisper sometimes hallucinates repeating phrases; collapse them.
def collapse_repeats(text):
    # Collapse immediate word/phrase repetitions like "que la verdad que la verdad..."
    words = text.split()
    if len(words) < 4:
        return text
    # Detect repeats of n-grams up to 6 words
    for n in range(2, 7):
        out = []
        i = 0
        while i < len(words):
            out.append(words[i])
            # If next n-gram matches previous n-gram, skip
            if i + n <= len(words) and i - n + 1 >= 0:
                if words[i - n + 1:i + 1] == words[i + 1:i + 1 + n]:
                    # skip the repetition
                    j = i + 1
                    while j + n <= len(words) and words[j - n + 1:j + 1] == words[j + 1:j + 1 + n]:
                        j += n
                    i = j + 1
                    continue
            i += 1
        words = out
    return " ".join(words)

# Build cleaned, non-overlapping segments.
clean_segs = []
seen_text = ""
for seg in segments:
    text = collapse_repeats(seg["text"]).strip()
    if not text or text.lower() in ("(música)", "[música]"):
        continue
    # If this segment overlaps with prior, advance start
    start = seg["start"]
    if clean_segs and start < clean_segs[-1]["end"]:
        start = clean_segs[-1]["end"] + 0.05
    end = seg["end"]
    if end <= start:
        end = start + 0.5
    # Drop trailing identical part with previous segment
    if clean_segs:
        prev = clean_segs[-1]["text"]
        # remove leading words that repeat the tail of prev
        prev_words = prev.split()
        cur_words = text.split()
        max_overlap = min(8, len(prev_words), len(cur_words))
        cut = 0
        for k in range(max_overlap, 0, -1):
            if [w.lower().strip(",.¿?¡!:;") for w in prev_words[-k:]] == [w.lower().strip(",.¿?¡!:;") for w in cur_words[:k]]:
                cut = k
                break
        if cut:
            cur_words = cur_words[cut:]
            text = " ".join(cur_words)
    if text:
        clean_segs.append({"start": start, "end": end, "text": text})

print(f"Cleaned: {len(clean_segs)} segments")

# Now split each segment into "subtitle lines" of max 6 words for nice display.
def split_into_lines(text, max_words=6):
    words = text.split()
    lines = []
    cur = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words and (w.endswith(',') or w.endswith('.') or w.endswith('?') or w.endswith('!') or len(cur) >= max_words + 2):
            lines.append(" ".join(cur))
            cur = []
    if cur:
        lines.append(" ".join(cur))
    return lines

# Build word-timed events
events = []
for seg in clean_segs:
    text = seg["text"]
    s, e = seg["start"], seg["end"]
    lines = split_into_lines(text, max_words=5)
    if not lines:
        continue
    seg_dur = e - s
    # Distribute lines proportional to word count
    total_words = sum(len(l.split()) for l in lines)
    if total_words == 0:
        continue
    cur_t = s
    for line in lines:
        wcount = len(line.split())
        line_dur = seg_dur * (wcount / total_words)
        line_dur = max(line_dur, 0.6)
        line_start = cur_t
        line_end = cur_t + line_dur
        if line_end > e:
            line_end = e
        events.append({
            "start": line_start,
            "end": line_end,
            "text": line,
            "words": line.split(),
        })
        cur_t = line_end

# Adjust last event end to <= duration
for ev in events:
    if ev["end"] > duration:
        ev["end"] = duration

# Generate ASS file
def t2ass(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    cs = int((s - int(s)) * 100)
    return f"{h:d}:{m:02d}:{int(s):02d}.{cs:02d}"

# Style: bold, thick black outline, white fill, big, centered near bottom
header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sub, Montserrat,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,5,2,2,40,40,90,1
Style: SubHL, Montserrat,64,&H0033FFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,5,2,2,40,40,90,1
Style: TitleBig, Montserrat,96,&H00FFFFFF,&H000099FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,3,5,40,40,40,1
Style: TitleSub, Montserrat,42,&H0033FFFF,&H000099FF,&H00000000,&H80000000,1,1,0,0,100,100,0,0,1,4,2,5,40,40,40,1
Style: Outro, Montserrat,84,&H00FFFFFF,&H000099FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,3,5,40,40,40,1
Style: TopBanner, Montserrat,38,&H0000FFFF,&H000099FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,8,40,40,30,1
Style: Bullet, Montserrat,38,&H00FFFFFF,&H000099FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,2,4,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

ass_events = []

# Helper to escape ass text
def esc(s):
    return s.replace("\\", "\\\\").replace("{", "(").replace("}", ")")

# Word-by-word karaoke style: each word appears with a pop animation
# We'll show the full line with the current word highlighted in yellow with a slight scale-up.
# To keep it readable, show the line for its full duration with karaoke fill highlighting word-by-word.

for ev in events:
    s, e = ev["start"], ev["end"]
    words = ev["words"]
    if not words:
        continue
    line_dur = e - s
    if line_dur <= 0:
        continue
    # \k tags use centiseconds. We'll do per-word durations evenly weighted by character count
    char_total = sum(len(w) for w in words) or 1
    # \kf for fill karaoke effect
    karaoke_text = ""
    for w in words:
        wd = max(15, int(line_dur * 100 * (len(w) / char_total)))
        karaoke_text += f"{{\\kf{wd}}}{esc(w)} "
    karaoke_text = karaoke_text.strip()
    # animation: pop in (scale from 70% to 100%) and slight bounce, drop shadow.
    # Order matters: tags after \t clobber animated values, so set initial state
    # before \t.
    pre = (
        "{\\fad(150,120)"
        "\\1c&H33FFFF&\\2c&HFFFFFF&\\3c&H000000&\\bord5\\shad2\\b1"
        "\\fscx70\\fscy70\\t(0,200,\\fscx100\\fscy100)}"
    )
    txt = pre + karaoke_text
    ass_events.append(
        f"Dialogue: 0,{t2ass(s)},{t2ass(e)},Sub,,0,0,0,,{txt}"
    )

# Add a top banner that pulses with show title
TITLE = "ESCUCHA ESTA HISTORIA"
banner_events = [
    (0.5, duration - 0.2, TITLE),
]
for s, e, t in banner_events:
    txt = f"{{\\fad(400,300)\\t(0,800,\\fscx105\\fscy105)\\an8\\pos(640,55)\\bord3\\shad2\\1c&H33FFFF&\\3c&H000000&\\b1}}{esc(t)}"
    ass_events.append(
        f"Dialogue: 0,{t2ass(s)},{t2ass(e)},TopBanner,,0,0,0,,{txt}"
    )

# Add periodic emoji/decoration lines (left/right floating shapes via text)
import random
random.seed(7)
deco_emojis = ["★", "✦", "✧", "♥", "✿", "❀", "✪", "⚡"]
for i in range(0, int(duration), 3):
    s = i + random.uniform(0, 1.0)
    e = s + random.uniform(1.5, 2.5)
    if e > duration:
        e = duration
    side = random.choice(["L", "R"])
    x = 80 if side == "L" else 1200
    y = random.randint(160, 560)
    color = random.choice(["&H33FFFF&", "&HFF99FF&", "&H99FFFF&", "&HFFFF66&", "&H66CCFF&"])
    rot = random.choice([-25, -10, 10, 25])
    emoji = random.choice(deco_emojis)
    txt = f"{{\\an5\\pos({x},{y})\\frz{rot}\\fs60\\1c{color}\\3c&H000000&\\bord3\\fad(300,300)\\t(0,400,\\fscx140\\fscy140)\\t(400,800,\\fscx100\\fscy100)}}{emoji}"
    ass_events.append(
        f"Dialogue: 1,{t2ass(s)},{t2ass(e)},Sub,,0,0,0,,{txt}"
    )

with open("subs.ass", "w", encoding="utf-8") as f:
    f.write(header)
    f.write("\n".join(ass_events))
    f.write("\n")
print(f"Wrote subs.ass with {len(ass_events)} events.")

# Save cleaned segments info for later
with open("clean_segments.json", "w", encoding="utf-8") as f:
    json.dump({"segments": clean_segs, "events": events, "duration": duration}, f, ensure_ascii=False, indent=2)
