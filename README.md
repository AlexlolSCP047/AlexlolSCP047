# Editor de Vídeo Profesional Animado

Pipeline en Python + ffmpeg que convierte un vídeo bruto en una pieza estilo
YouTube con intro animada, fondo desenfocado, subtítulos karaoke palabra a
palabra, banner superior, decoraciones flotantes, barra de progreso y outro
con llamadas a la acción.

## Salida

`final.mp4` (1280×720, 30 fps, ~66 s):

1. **Intro (3 s)** — `ESCUCHA / ESTA HISTORIA` con zoom-in, gradiente animado
   de color, estrellas giratorias, hashtags, tagline italica.
2. **Cuerpo (~59,5 s)** — vídeo original (vertical) compuesto sobre un fondo
   desenfocado y escalado del propio vídeo, recorte centrado con borde cian,
   color grading (saturación + contraste + sharpen + viñeta), barra de
   progreso animada, banner superior pulsante, emojis decorativos
   aleatorios y subtítulos en karaoke (\\kf) blancos que se rellenan en
   amarillo conforme se pronuncian las palabras.
3. **Outro (4 s)** — `¡GRACIAS / POR VER!` con pop-in, llamadas a la acción
   `♥ ME GUSTA` y `▶ SUSCRIBIRSE`, decoraciones rotando en las cuatro
   esquinas y gradiente animado.

## Pipeline

```
input.mp4 ──┐
            │ ffmpeg → audio.wav (16k mono PCM)
            │
            └─► transcribe_sherpa.py
                    │ sherpa-onnx whisper-small
                    └─► segments.json
                            │
                            └─► build_subs.py
                                    │ limpieza de repeticiones,
                                    │ split en líneas, karaoke por palabra
                                    └─► subs.ass
                                            │
input.mp4 ──────────────────────────────────┴─► render.sh
                                                  │ intro + main + outro
                                                  └─► final.mp4
```

## Requisitos

- ffmpeg con `libass` (provisto por Ubuntu).
- `sherpa-onnx` (PyPI) + el modelo `sherpa-onnx-whisper-small` (~640 MB)
  desde [github.com/k2-fsa/sherpa-onnx releases][rel] descomprimido en
  `models/`.
- Fuente Montserrat (`fonts-montserrat`).

[rel]: https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

## Cómo regenerar

```bash
# 1) Audio
ffmpeg -y -i input.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav

# 2) Transcripción
python3 transcribe_sherpa.py

# 3) ASS animado
python3 build_subs.py

# 4) Render final
bash render.sh
```

## Detalles técnicos

- **Karaoke**: cada palabra de cada línea recibe un `\kfNN` proporcional a su
  longitud, sobre un estilo con `PrimaryColour=amarillo` y
  `SecondaryColour=blanco`, de modo que las palabras pronunciadas se rellenan
  en amarillo.
- **Pop-in por línea**: `\fscx70\fscy70\t(0,200,\fscx100\fscy100)` para que
  cada línea aparezca con un rebote sutil.
- **Composición del cuerpo**: `split` del original → fondo escalado a 1280×720
  con `gblur=sigma=22` + `eq` para oscurecer + `hue` ciclando, y primer plano
  sin recortar, escalado a 720 px de alto y bordeado con cian semi-translúcido.
- **Audio**: `loudnorm=I=-16:LRA=11:TP=-1.5` + `acompressor` + `highpass=70`
  para quitar zumbido y normalizar a estándar de YouTube.
- **Transiciones intro/outro**: `blend=all_expr` entre dos `color=` lavfi para
  un degradado animado, `hue` rotando en el tiempo y `gblur` para un look
  cinematográfico.
