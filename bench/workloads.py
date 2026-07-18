"""Workload generators: calibrated inputs whose token counts are known in advance.

Every "cold" measurement needs a unique input (so the server's caches can
never silently warm it) with an exactly known token count (so the x-axis of
the TTFT curves is trusted before the server confirms it).
"""

import base64
import io

import numpy as np
from PIL import Image


def image_tokens(width: int, height: int) -> int:
    """Tokens an image costs in Qwen2-VL: one per 28x28 pixel block.

    28 = the vision transformer's 14px patch size x a 2x2 patch merge.
    Width and height count independently (like a conv feature map), so a
    896x448 image is 32 x 16 = 512 tokens -- no resizing to square happens.
    Qwen-specific: Phi-3.5-vision has no such closed form (it uses crops).
    """
    return (height // 28) * (width // 28)


def noise_image(side_px: int, seed: int) -> Image.Image:
    """A square RGB random-noise image.

    Noise, not photos: serving cost depends on tensor shapes, not pixel
    values, and noise is generatable at any exact size. A distinct seed
    gives a distinct image, so cold requests can never be cache-warm
    (vLLM hashes image bytes).
    """
    if side_px % 28 != 0:
        # sides must land exactly on the 28px token grid, otherwise the
        # server's smart_resize would change the size and break our
        # predicted token count
        raise ValueError(f"side_px must be a multiple of 28, got {side_px}")
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(side_px, side_px, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def to_data_url(img: Image.Image) -> str:
    """Encode an image as the data-URL string the OpenAI-style API expects.

    The image travels inside the request JSON itself: JPEG bytes -> base64
    text -> "data:image/jpeg;base64,...." (no separate file upload).
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def text_prompt(target_tokens: int, tokenizer, seed: int) -> str:
    """A unique string that tokenizes to exactly target_tokens tokens
    (counted without special tokens; the chat template adds its own ~19).

    Recipe: build over-long filler text, cut the TOKEN-ID list (not the
    string -- characters don't map 1:1 to tokens) to the target length,
    decode back to a string. Then repair: decode->encode is not a perfect
    round trip, so re-count and nudge until exact.
    The "case {seed}:" prefix makes every prompt unique (defeats the
    prefix cache) without changing the token count target.
    """
    text = f"case {seed}:" + " benchmark" * target_tokens
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < target_tokens:
        raise ValueError(f"filler too short for {target_tokens} tokens")
    text = tokenizer.decode(ids[:target_tokens])

    for _ in range(50):  # bounded repair loop; typically converges in 0-2 steps
        count = len(tokenizer.encode(text, add_special_tokens=False))
        if count == target_tokens:
            return text
        if count > target_tokens:
            ids = tokenizer.encode(text, add_special_tokens=False)[:-1]
            text = tokenizer.decode(ids)
        else:
            text += " ok"
    raise RuntimeError(f"could not converge on {target_tokens} tokens")


def text_payload(model: str, prompt: str, max_tokens: int = 64) -> dict:
    """Request body for a text-only chat completion.

    temperature=0 makes generation deterministic: the same input always
    yields the same output, so runs are reproducible.
    """
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def image_payload(model: str, data_url: str, question: str, max_tokens: int = 64,
                  image_first: bool = False) -> dict:
    """Request body for an image + question chat completion.

    image_first matters for prefix caching: the cache reuses a shared
    *prefix*, so repeated content (the image) must come before the part
    that changes (the question) or no request can ever hit the cache.
    """
    text_block = {"type": "text", "text": question}
    image_block = {"type": "image_url", "image_url": {"url": data_url}}
    content = [image_block, text_block] if image_first else [text_block, image_block]
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
