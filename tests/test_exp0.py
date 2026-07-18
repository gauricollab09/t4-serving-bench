from bench.exp0 import looks_coherent
from bench.workloads import image_payload, text_payload


def test_real_caption_is_coherent():
    assert looks_coherent("A colorful field of random static noise, like a TV screen.")


def test_gibberish_is_not():
    assert not looks_coherent("!!!!!!!!!!!!!!!!")
    assert not looks_coherent("")
    assert not looks_coherent("aaaaaaaaaaaaaaaaaa")


def test_payload_shapes():
    t = text_payload("m", "hi")
    assert t["messages"][0]["content"] == "hi"
    assert t["temperature"] == 0

    i = image_payload("m", "data:image/jpeg;base64,xyz", "what is this?")
    blocks = i["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "image_url"]
