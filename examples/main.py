from pathlib import Path

from rkllm_runtime.rkllm import RKLLM


models_dir = Path(__file__).resolve(strict=True).parent / "models"
llama_32_3b = (
    models_dir / "Llama-3.2-3B-Instruct-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm"
)

with RKLLM(model_path=bytes(llama_32_3b)) as rkllm:
    rkllm.run(prompt=b"Hello, world!")
