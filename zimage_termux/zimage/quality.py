"""Locked quality preset for the `Zimage` command.

The negative prompt and guidance defaults are intentionally constants in code
(not flags) so every invocation produces consistent high-quality output.
Editing this file is the only way to change them.
"""

from __future__ import annotations

from typing import Final


# Non-editable from the CLI. To change, edit this string and reinstall.
NEGATIVE_PROMPT: Final[str] = (
    "low quality, lowres, worst quality, blurry, out of focus, jpeg artifacts, "
    "compression artifacts, oversharpened, oversaturated, undersaturated, "
    "monochrome, grayscale, washed out, noisy, grainy, banding, posterization, "
    "deformed, disfigured, mutated, mutation, bad anatomy, extra limbs, "
    "missing limbs, extra fingers, missing fingers, fused fingers, malformed hands, "
    "asymmetric eyes, cross-eyed, watermark, signature, text, logo, caption, "
    "stock photo border, ugly, duplicate, cropped, cut off, draft, sketchy, "
    "amateur, low detail, plastic skin, waxy skin, cartoonish when photoreal"
)

# Baked-in defaults for `Zimage [prompt]` runs. CLI flags can still override
# size/seed/out, but everything else is locked.
DEFAULT_STEPS: Final[int] = 8
DEFAULT_SIZE: Final[int] = 1024
DEFAULT_GUIDANCE_SCALE: Final[float] = 2.0
