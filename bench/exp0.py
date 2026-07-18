"""EXP-0: the fp16 correctness gate.

Qwen2-VL was trained in bf16; the T4 GPU only supports fp16. If fp16 breaks
the model, outputs degrade to gibberish ("!!!!"), empty text, or the server
crashes. Nothing else in this project is worth running until this passes.

Pass = a coherent caption AND a prompt_tokens count that matches the
28-px-tile formula. Fail = switch to the Phi-3.5-vision fallback.
"""

from bench.client import send_request
from bench.workloads import image_payload, image_tokens, noise_image, to_data_url

MODEL = "Qwen/Qwen2-VL-2B-Instruct"


def looks_coherent(text: str) -> bool:
    """Cheap gibberish detector: a real caption is non-trivial, varied text."""
    stripped = text.strip()
    return len(stripped) >= 10 and "!!!!" not in stripped and len(set(stripped)) > 5


def run_gate(base_url: str = "http://localhost:8000") -> dict:
    side = 448
    image = noise_image(side, seed=0)
    payload = image_payload(MODEL, to_data_url(image), "Describe this image.")
    result = send_request(payload, base_url)

    # what we sent: 256 image tokens + 2 vision delimiters, plus the chat
    # template and the question text (roughly 20-40 more tokens)
    vision_tokens = image_tokens(side, side) + 2
    extra = (result["prompt_tokens"] or 0) - vision_tokens

    result["vision_tokens_expected"] = vision_tokens
    result["coherent"] = looks_coherent(result["output_text"])
    result["token_formula_ok"] = 0 < extra <= 40
    result["gate_passed"] = result["coherent"] and result["token_formula_ok"]
    return result


if __name__ == "__main__":
    from pprint import pprint

    pprint(run_gate())
