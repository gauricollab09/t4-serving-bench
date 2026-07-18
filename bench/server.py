"""Launch and stop the vLLM server with named flag configurations.

EXP-C needs the same model served twice: once with automatic prefix
caching disabled and once with the default (enabled). Each config is a
fresh process so no cache state leaks between configurations.
"""

import subprocess
import time

import requests

MODEL = "Qwen/Qwen2-VL-2B-Instruct"


QWEN_MM_KWARGS = '{"min_pixels": 784, "max_pixels": 1003520}'
PHI_MM_KWARGS = '{"num_crops": 16}'


def build_args(model: str = MODEL, apc: bool = True, port: int = 8000,
               mm_kwargs: str = QWEN_MM_KWARGS, gpu_mem: float = 0.85,
               trust_remote_code: bool = False) -> list:
    # Flag rationale (all verified on a Colab T4):
    #   --dtype half            T4 has no bf16 (compute capability 7.5 < 8)
    #   --max-model-len 4096    cap context so the KV cache fits 16 GB
    #   --gpu-memory-utilization headroom; vLLM pre-allocates at startup
    #   --enforce-eager         no CUDA graphs (vision encoders have dynamic
    #                           shapes; capture can fail) + saves memory
    #   --limit-mm-per-prompt / --mm-processor-kwargs
    #                           bound the worst-case image so the startup
    #                           memory profiling pass doesn't OOM
    args = [
        "vllm", "serve", model,
        "--dtype", "half",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", str(gpu_mem),
        "--enforce-eager",
        "--limit-mm-per-prompt", '{"image": 1}',
        "--mm-processor-kwargs", mm_kwargs,
        "--port", str(port),
    ]
    if trust_remote_code:
        args.append("--trust-remote-code")
    if not apc:
        args.append("--no-enable-prefix-caching")
    return args


def start(model: str = MODEL, apc: bool = True, port: int = 8000,
          log_path: str = "/content/vllm.log", **kwargs):
    log_file = open(log_path, "w")
    return subprocess.Popen(build_args(model, apc, port, **kwargs),
                            stdout=log_file, stderr=subprocess.STDOUT)


def wait_ready(proc, base_url: str = "http://localhost:8000",
               log_path: str = "/content/vllm.log", timeout_s: int = 600) -> None:
    for _ in range(timeout_s // 5):
        if proc.poll() is not None:
            raise RuntimeError("server exited:\n" + open(log_path).read()[-3000:])
        try:
            if requests.get(f"{base_url}/v1/models", timeout=2).ok:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
    raise RuntimeError(f"server not ready within {timeout_s}s")


def stop(proc) -> None:
    proc.terminate()
    proc.wait(timeout=30)
