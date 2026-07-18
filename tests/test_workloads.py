import base64

import pytest
from transformers import AutoTokenizer

from bench.workloads import image_tokens, noise_image, text_prompt, to_data_url

SWEEP = {224: 64, 448: 256, 672: 576, 896: 1024}


def test_448_square_is_256_tokens():
    # 448/28 = 16 blocks per side -> 16*16 tokens
    assert image_tokens(448, 448) == 256


def test_all_sweep_sizes():
    for side, tokens in SWEEP.items():
        assert image_tokens(side, side) == tokens


def test_non_square_image():
    # 32 blocks across, 16 blocks down
    assert image_tokens(896, 448) == 512


def test_noise_image_dimensions_and_mode():
    img = noise_image(224, seed=0)
    assert img.size == (224, 224)
    assert img.mode == "RGB"


def test_noise_image_rejects_non_multiple_of_28():
    with pytest.raises(ValueError):
        noise_image(447, seed=0)


def test_noise_images_are_unique_per_seed():
    a = noise_image(224, seed=1)
    b = noise_image(224, seed=2)
    assert a.tobytes() != b.tobytes()


def test_data_url_shape_and_decodability():
    url = to_data_url(noise_image(224, seed=0))
    prefix = "data:image/jpeg;base64,"
    assert url.startswith(prefix)
    raw = base64.b64decode(url[len(prefix):])
    assert raw[:2] == b"\xff\xd8"  # JPEG magic bytes


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


@pytest.mark.parametrize("target", [64, 256, 576, 1024])
def test_text_prompt_exact_token_count(tokenizer, target):
    prompt = text_prompt(target, tokenizer, seed=1)
    assert len(tokenizer.encode(prompt, add_special_tokens=False)) == target


def test_text_prompts_unique_per_seed(tokenizer):
    assert text_prompt(64, tokenizer, seed=1) != text_prompt(64, tokenizer, seed=2)


def test_image_payload_block_order():
    from bench.workloads import image_payload
    default = image_payload("m", "data:x", "q")
    first = image_payload("m", "data:x", "q", image_first=True)
    assert [b["type"] for b in default["messages"][0]["content"]] == ["text", "image_url"]
    assert [b["type"] for b in first["messages"][0]["content"]] == ["image_url", "text"]
