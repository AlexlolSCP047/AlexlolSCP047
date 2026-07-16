#!/bin/bash
# Professional YouTube-style video render pipeline.
# Stages:
#   1. Render an animated intro (3s) using ASS overlay on a moving gradient.
#   2. Process the main clip: scale to 1280x720 with pillar-/letter-boxing,
#      apply Ken Burns zoom-pan, color grade, animated bottom progress bar,
#      side decorations, and burn-in karaoke ASS subtitles.
#   3. Render an animated outro (4s) with thanks/like-subscribe ASS overlay.
#   4. Concatenate intro + main + outro into final.mp4.

set -euo pipefail

INPUT="input.mp4"
W=1280
H=720
FPS=30

# Probe duration of source
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT")
echo "Source duration: $DUR"

# ---------- 1) INTRO (3 seconds) ----------
# Animated gradient + pop-in title via ASS.
cat > intro.ass <<'EOF'
[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TitleBig, Montserrat,110,&H00FFFFFF,&H000099FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,8,4,5,40,40,40,1
Style: TitleSub, Montserrat,52,&H0033FFFF,&H000099FF,&H00000000,&H80000000,1,1,0,0,100,100,0,0,1,4,2,5,40,40,40,1
Style: TitleHash, Montserrat,40,&H00FFFFFF,&H000099FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,2,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:03.00,TitleBig,,0,0,0,,{\an5\pos(640,290)\fad(200,200)\bord10\shad6\1c&HFFFFFF&\3c&H1A1A1A&\fscx20\fscy20\frz-25\t(0,500,\fscx100\fscy100\frz0)}ESCUCHA\NESTA HISTORIA
Dialogue: 0,0:00:00.50,0:00:03.00,TitleSub,,0,0,0,,{\an5\pos(640,500)\fad(400,200)\bord4\shad2\1c&H33FFFF&\3c&H000000&\b1}— UN CUENTO PARA NO OLVIDAR —
Dialogue: 0,0:00:00.80,0:00:03.00,TitleHash,,0,0,0,,{\an5\pos(640,590)\fad(500,200)\bord3\1c&HFFFFFF&\3c&H000000&}#historias #abuelos #cuentos
Dialogue: 1,0:00:00.00,0:00:03.00,TitleBig,,0,0,0,,{\an5\pos(120,360)\frz25\fad(0,400)\t(0,2800,\frz385)\fs220\1c&HFFCC33&\3c&H000000&\bord4\alpha&H99&}★
Dialogue: 1,0:00:00.00,0:00:03.00,TitleBig,,0,0,0,,{\an5\pos(1160,360)\frz-15\fad(0,400)\t(0,2800,\frz-375)\fs220\1c&HFF66CC&\3c&H000000&\bord4\alpha&H99&}✦
Dialogue: 2,0:00:00.00,0:00:03.00,TitleHash,,0,0,0,,{\an5\pos(640,140)\fs60\1c&HFFFFFF&\3c&H000000&\bord3\fad(200,200)\t(0,2800,\fscx115\fscy115)}▶ PRESENTA
EOF

# Animated gradient background using ffmpeg (color cycle).
ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "color=c=0x0c1a2b:s=${W}x${H}:r=${FPS}:d=3" \
  -f lavfi -i "color=c=0x2a004c:s=${W}x${H}:r=${FPS}:d=3" \
  -f lavfi -i "color=c=0x004c66:s=${W}x${H}:r=${FPS}:d=3" \
  -f lavfi -i "anullsrc=channel_layout=stereo:sample_rate=44100:d=3" \
  -filter_complex "
    [0:v][1:v]blend=all_expr='A*(1-T/3)+B*(T/3)':shortest=1[bg1];
    [bg1][2:v]blend=all_expr='A*(if(lte(T,1.5),1,1-(T-1.5)/1.5))+B*(if(lte(T,1.5),0,(T-1.5)/1.5))':shortest=1[bg];
    [bg]format=yuv420p,
       drawbox=x=0:y=0:w=iw:h=ih:color=black@0.0:t=fill,
       hue=h='30+t*60':s=1.2,
       gblur=sigma=4,
       eq=brightness=-0.05:contrast=1.05,
       ass=intro.ass[v]
  " \
  -map "[v]" -map 3:a \
  -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 20 \
  -c:a aac -b:a 192k -shortest \
  intro.mp4
echo "intro.mp4 done"

# ---------- 2) OUTRO (4 seconds) ----------
cat > outro.ass <<'EOF'
[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Big, Montserrat,128,&H00FFFFFF,&H000099FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,10,4,5,40,40,40,1
Style: Mid, Montserrat,52,&H0033FFFF,&H000099FF,&H00000000,&H80000000,1,1,0,0,100,100,0,0,1,5,2,5,40,40,40,1
Style: Btn, Montserrat,42,&H00FFFFFF,&H000099FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,2,5,40,40,40,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:04.00,Big,,0,0,0,,{\an5\pos(640,250)\fad(200,300)\bord12\shad6\1c&HFFFFFF&\3c&H1A1A1A&\fscx40\fscy40\t(0,600,\fscx100\fscy100)}¡GRACIAS\NPOR VER!
Dialogue: 0,0:00:00.50,0:00:04.00,Mid,,0,0,0,,{\an5\pos(640,470)\fad(400,300)\bord5\shad2\1c&H33FFFF&\3c&H000000&\b1}Si te gustó, suscríbete
Dialogue: 0,0:00:00.90,0:00:04.00,Btn,,0,0,0,,{\an5\pos(380,580)\fad(500,300)\bord4\shad3\1c&HFFFFFF&\3c&H3030E0&\b1\bord3}♥  ME GUSTA
Dialogue: 0,0:00:01.20,0:00:04.00,Btn,,0,0,0,,{\an5\pos(900,580)\fad(500,300)\bord4\shad3\1c&HFFFFFF&\3c&H1010CC&\b1\bord3}▶  SUSCRIBIRSE
Dialogue: 2,0:00:00.00,0:00:04.00,Big,,0,0,0,,{\an5\pos(140,140)\fs150\1c&HFFCC33&\3c&H000000&\bord4\alpha&H77&\frz-20\fad(300,300)\t(0,3700,\frz340)}★
Dialogue: 2,0:00:00.00,0:00:04.00,Big,,0,0,0,,{\an5\pos(1140,140)\fs150\1c&HFF66CC&\3c&H000000&\bord4\alpha&H77&\frz20\fad(300,300)\t(0,3700,\frz-340)}✦
Dialogue: 2,0:00:00.00,0:00:04.00,Big,,0,0,0,,{\an5\pos(140,640)\fs150\1c&H66CCFF&\3c&H000000&\bord4\alpha&H77&\frz30\fad(300,300)\t(0,3700,\frz-330)}✿
Dialogue: 2,0:00:00.00,0:00:04.00,Big,,0,0,0,,{\an5\pos(1140,640)\fs150\1c&H99FF66&\3c&H000000&\bord4\alpha&H77&\frz-30\fad(300,300)\t(0,3700,\frz330)}✪
EOF

ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "color=c=0x140033:s=${W}x${H}:r=${FPS}:d=4" \
  -f lavfi -i "color=c=0x4c0080:s=${W}x${H}:r=${FPS}:d=4" \
  -f lavfi -i "anullsrc=channel_layout=stereo:sample_rate=44100:d=4" \
  -filter_complex "
    [0:v][1:v]blend=all_expr='A*(0.5+0.5*sin(2*PI*T/4))+B*(0.5-0.5*sin(2*PI*T/4))':shortest=1[bg];
    [bg]format=yuv420p,hue=h='t*30':s=1.3,gblur=sigma=3,ass=outro.ass[v]
  " \
  -map "[v]" -map 2:a \
  -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 20 \
  -c:a aac -b:a 192k -shortest \
  outro.mp4
echo "outro.mp4 done"

# ---------- 3) MAIN: Apply effects + subs ----------
# - Source is portrait 576x1024 (after auto-rotation). Build a YouTube-style
#   landscape composition: large blurred background + sharp foreground portrait.
# - Center foreground portrait scaled to 405x720 with rounded-corner mask via
#   geq/format isn't needed; we use a colored border drawbox instead.
# - Color grade + sharpen + vignette + animated progress bar on bottom.
# - Burn-in karaoke subtitles via ass=subs.ass (1280x720 PlayRes).
# - Audio: loudnorm + compressor + highpass.

ffmpeg -y -hide_banner -loglevel error \
  -i "$INPUT" \
  -filter_complex "
    [0:v]fps=${FPS},setsar=1,split=2[src1][src2];

    [src1]scale=${W}:${H}:force_original_aspect_ratio=increase,
          crop=${W}:${H},
          eq=brightness=-0.18:saturation=1.25:contrast=0.95,
          gblur=sigma=22,
          hue=h='t*8'[bg];

    [src2]scale=-2:${H},
          eq=contrast=1.08:saturation=1.18:brightness=0.02:gamma=0.97,
          unsharp=lx=5:ly=5:la=0.7,
          pad=iw+12:ih+12:6:6:color=0x33FFFFAA[fg];

    [bg][fg]overlay=x=(W-w)/2:y=(H-h)/2:format=auto[comp];

    [comp]vignette=PI/4.8,
          drawbox=x=0:y=ih-14:w=iw*t/${DUR}:h=14:color=0x33FFFF@0.95:t=fill,
          drawbox=x=0:y=ih-14:w=iw:h=14:color=0xFFFFFF@0.25:t=2,
          drawbox=x=0:y=0:w=iw:h=80:color=0x000000@0.40:t=fill,
          drawbox=x=0:y=ih-80:w=iw:h=66:color=0x000000@0.50:t=fill[boxed];

    [boxed]ass=subs.ass,format=yuv420p[v]
  " \
  -filter_complex_threads 4 \
  -af "loudnorm=I=-16:LRA=11:TP=-1.5,acompressor=threshold=-18dB:ratio=3:attack=10:release=200,highpass=f=70,aresample=44100" \
  -map "[v]" -map 0:a \
  -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 20 \
  -c:a aac -b:a 192k \
  main.mp4
echo "main.mp4 done"

# ---------- 4) Concatenate ----------
# Re-encode concat for safety with consistent codecs/timebase.
ffmpeg -y -hide_banner -loglevel error \
  -i intro.mp4 -i main.mp4 -i outro.mp4 \
  -filter_complex "
    [0:v]setsar=1,fps=${FPS},format=yuv420p[v0];
    [1:v]setsar=1,fps=${FPS},format=yuv420p[v1];
    [2:v]setsar=1,fps=${FPS},format=yuv420p[v2];
    [v0][0:a][v1][1:a][v2][2:a]concat=n=3:v=1:a=1[v][a]
  " \
  -map "[v]" -map "[a]" \
  -c:v libx264 -pix_fmt yuv420p -preset medium -crf 20 \
  -c:a aac -b:a 192k -movflags +faststart \
  final.mp4
echo "final.mp4 done"

ls -la intro.mp4 main.mp4 outro.mp4 final.mp4
