from bench.server import build_args


def test_apc_flag_toggles():
    assert "--no-enable-prefix-caching" not in build_args(apc=True)
    assert "--no-enable-prefix-caching" in build_args(apc=False)


def test_core_t4_flags_present():
    args = build_args()
    for flag in ("--dtype", "--enforce-eager", "--max-model-len"):
        assert flag in args


def test_other_model_options():
    from bench.server import PHI_MM_KWARGS, build_args
    args = build_args(model="microsoft/Phi-3.5-vision-instruct",
                      mm_kwargs=PHI_MM_KWARGS, gpu_mem=0.9,
                      trust_remote_code=True)
    assert "microsoft/Phi-3.5-vision-instruct" in args
    assert "--trust-remote-code" in args
    assert '{"num_crops": 16}' in args
    assert "0.9" in args
