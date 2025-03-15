from typing import Any, Callable

import _cffi_backend

class RKLLMLib:
    def __dir__(self): ...

    # LLMCallState
    RKLLM_RUN_NORMAL: int
    RKLLM_RUN_WAITING: int
    RKLLM_RUN_FINISH: int
    RKLLM_RUN_ERROR: int
    RKLLM_RUN_GET_LAST_HIDDEN_LAYER: int

    # RKLLMInputType
    RKLLM_INPUT_PROMPT: int
    RKLLM_INPUT_TOKEN: int
    RKLLM_INPUT_EMBED: int
    RKLLM_INPUT_MULTIMODAL: int

    # RKLLMInferMode
    RKLLM_INFER_GENERATE: int
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER: int

    rkllm_result_callback: Callable[[Any, Any, int], Any]

    rkllm_createDefaultParam: Callable[[], Any]

    rkllm_init: Callable[[Any, Any, Any], int]

    rkllm_load_lora: Callable[[Any, Any], int]

    rkllm_load_prompt_cache: Callable[[Any, Any], int]

    rkllm_release_prompt_cache: Callable[[Any], int]

    rkllm_destroy: Callable[[Any], int]

    rkllm_run: Callable[[Any, Any, Any, Any], int]

    rkllm_run_async: Callable[[Any, Any, Any, Any], int]

    rkllm_abort: Callable[[Any], int]

    rkllm_is_running: Callable[[Any], int]

ffi: _cffi_backend.FFI
lib: RKLLMLib
