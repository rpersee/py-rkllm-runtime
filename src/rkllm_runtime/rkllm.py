from enum import IntEnum
from typing import ClassVar, Any, overload

from ._rkllm import ffi, lib
from .utils import CStructMeta, from_dict


class LLMCallState(IntEnum):
    """Describes the possible states of an LLM call."""

    NORMAL = lib.RKLLM_RUN_NORMAL
    """The LLM call is in a normal running state."""
    WAITING = lib.RKLLM_RUN_WAITING
    """The LLM call is waiting for complete UTF-8 encoded character."""
    FINISH = lib.RKLLM_RUN_FINISH
    """The LLM call has finished execution."""
    ERROR = lib.RKLLM_RUN_ERROR
    """An error occurred during the LLM call."""
    GET_LAST_HIDDEN_LAYER = lib.RKLLM_RUN_GET_LAST_HIDDEN_LAYER
    """Retrieve the last hidden layer during inference."""


class RKLLMInputType(IntEnum):
    """Defines the types of inputs that can be fed into the LLM."""

    PROMPT = lib.RKLLM_INPUT_PROMPT
    """Input is a text prompt."""
    TOKEN = lib.RKLLM_INPUT_TOKEN
    """Input is a sequence of tokens."""
    EMBED = lib.RKLLM_INPUT_EMBED
    """Input is an embedding vector."""
    MULTIMODAL = lib.RKLLM_INPUT_MULTIMODAL
    """Input is multimodal (e.g., text and image)."""


class RKLLMInferMode(IntEnum):
    """Specifies the inference modes of the LLM."""

    GENERATE = lib.RKLLM_INFER_GENERATE
    """The LLM generates text based on input."""
    GET_LAST_HIDDEN_LAYER = lib.RKLLM_INFER_GET_LAST_HIDDEN_LAYER
    """The LLM retrieves the last hidden layer for further processing."""


class RKLLMExtendParam(metaclass=CStructMeta):
    """The extend parameters for configuring an LLM instance."""

    base_domain_id: int
    reserved: tuple[int, ...]


class RKLLMParam(metaclass=CStructMeta):
    """Defines the parameters for configuring an LLM instance."""

    model_path: bytes
    """Path to the model file."""
    max_context_len: int
    """Maximum number of tokens in the context window."""
    max_new_tokens: int
    """Maximum number of new tokens to generate."""
    top_k: int
    """Top-K sampling parameter for token generation."""
    top_p: float
    """Top-P (nucleus) sampling parameter."""
    temperature: float
    """Sampling temperature, affecting the randomness of token selection."""
    repeat_penalty: float
    """Penalty for repeating tokens in generation."""
    frequency_penalty: float
    """Penalizes frequent tokens during generation."""
    presence_penalty: float
    """Penalizes tokens based on their presence in the input."""
    mirostat: int
    """Mirostat sampling strategy flag (0 to disable)."""
    mirostat_tau: float
    """Tau parameter for Mirostat sampling."""
    mirostat_eta: float
    """Eta parameter for Mirostat sampling."""
    skip_special_token: bool
    """Whether to skip special tokens during generation."""
    is_async: bool
    """Whether to run inference asynchronously."""
    img_start: bytes
    """Starting position of an image in multimodal input."""
    img_end: bytes
    """Ending position of an image in multimodal input."""
    img_content: bytes
    """Pointer to the image content."""
    extend_param: RKLLMExtendParam
    """Extend parameters."""

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "max_context_len": 512,
        "max_new_tokens": -1,
        "skip_special_token": True,
        "top_k": 1,
        "top_p": 0.9,
        "temperature": 0.8,
        "repeat_penalty": 1.1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "mirostat": 0,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "is_async": False,
        "img_start": b"<img>",
        "img_end": b"</img>",
        "img_content": b"<unk>",
        "extend_param": {"base_domain_id": 0},
    }

    @classmethod
    def from_defaults(cls) -> "RKLLMParam":
        """Create an instance of the class with default values."""
        # getting defaults from lib.rkllm_createDefaultParam and
        # updating with new values leads to weird behavior
        return from_dict(cls, cls.DEFAULTS)


class RKLLMLoraAdapter(metaclass=CStructMeta):
    """Defines parameters for a Lora adapter used in model fine-tuning."""

    lora_adapter_path: bytes
    """Path to the Lora adapter file."""
    lora_adapter_name: bytes
    """Name of the Lora adapter."""
    scale: float
    """Scaling factor for applying the Lora adapter."""


class RKLLMEmbedInput(metaclass=CStructMeta):
    """Represents an embedding input to the LLM."""

    embed: tuple[float]
    """Pointer to the embedding vector (of size n_tokens * n_embed)."""
    n_tokens: int
    """Number of tokens represented in the embedding."""


class RKLLMTokenInput(metaclass=CStructMeta):
    """Represents token input to the LLM."""

    input_ids: tuple[int]
    """Array of token IDs."""
    n_tokens: int
    """Number of tokens in the input."""


class RKLLMMultiModelInput(metaclass=CStructMeta):
    """Represents multimodal input (e.g., text and image)."""

    prompt: bytes
    """Text prompt input."""
    image_embed: tuple[float]
    """Embedding of the image (of size n_image_tokens * n_image_embed)."""
    n_image_tokens: int
    """Number of image tokens."""


class RKLLMLoraParam(metaclass=CStructMeta):
    """Structure defining parameters for Lora adapters."""

    lora_adapter_name: bytes
    """Name of the Lora adapter."""


# class RKLLMInput(metaclass=CStructMeta): ...
# implemented as overloaded methods on RKLLM.run


class RKLLMPromptCacheParam(metaclass=CStructMeta):
    """Structure to define parameters for caching prompts."""

    save_prompt_cache: int
    """Flag to indicate whether to save the prompt cache (0 = don't save, 1 = save)."""
    prompt_cache_path: bytes
    """Path to the prompt cache file."""


class RKLLMInferParam(metaclass=CStructMeta):
    """Structure for defining parameters during inference."""

    mode: RKLLMInferMode
    """Inference mode (e.g., generate or get last hidden layer)."""
    lora_params: RKLLMLoraParam
    """Pointer to Lora adapter parameters."""
    prompt_cache_params: RKLLMPromptCacheParam
    """Pointer to prompt cache parameters."""


class RKLLMResultLastHiddenLayer(metaclass=CStructMeta):
    """Structure to hold the hidden states from the last layer."""

    hidden_states: tuple[float]
    """Pointer to the hidden states (of size num_tokens * embd_size)."""
    embd_size: int
    """Size of the embedding vector."""
    num_tokens: int
    """Number of tokens for which hidden states are stored."""


class RKLLMResult(metaclass=CStructMeta):
    """Structure to represent the result of LLM inference."""

    text: bytes
    """Generated text result."""
    token_id: int
    """ID of the generated token."""
    last_hidden_layer: RKLLMResultLastHiddenLayer
    """Hidden states of the last layer (if requested)."""


class RKLLM:
    def __init__(self, model_path: bytes, config: dict[str, Any] | None = None):
        if config is None:
            param = RKLLMParam.from_defaults()
        else:
            param = from_dict(RKLLMParam, config)
        param.model_path = ffi.new("char[]", model_path)  # type: ignore[assignment]

        # keep a reference to the function object to prevent GC
        self._callback_ref = ffi.def_extern(name="rkllm_result_callback")(
            self._result_callback
        )

        self._handle = ffi.new("LLMHandle *")
        rc = lib.rkllm_init(self._handle, param, lib.rkllm_result_callback)
        if rc != 0:
            raise RuntimeError(f"LLM init failed with error code {rc}")

    def _result_callback(
        self, result: RKLLMResult, userdata: None, state: LLMCallState
    ):
        """Callback function to handle LLM results.

        :param result: Pointer to the LLM result.
        :param userdata: Pointer to user data for the callback.
        :param state: State of the LLM call (e.g., finished, error).
        """
        if state == LLMCallState.NORMAL:
            print(ffi.string(result.text).decode(), end="", flush=True)  # type: ignore
        elif state == LLMCallState.FINISH:
            print()
        else:
            print(f"[Callback] State: {LLMCallState(state).name}")

    def load_lora(
        self, lora_adapter_path: bytes, lora_adapter_name: bytes, scale: float = 1.0
    ):
        """Load a Lora adapter into the LLM.

        :param lora_adapter_path: Path to the Lora adapter file.
        :param lora_adapter_name: Name of the Lora adapter.
        :param scale: Scaling factor for applying the Lora adapter.
        :raises RuntimeError: If loading the adapter fails.
        """
        lora_adapter = RKLLMLoraAdapter(
            lora_adapter_path=lora_adapter_path,
            lora_adapter_name=lora_adapter_name,
            scale=scale,
        )
        rc = lib.rkllm_load_lora(self._handle[0], lora_adapter)
        if rc != 0:
            raise RuntimeError(f"Failed to load Lora adapter, error code: {rc}")

    def load_prompt_cache(self, prompt_cache_path: bytes):
        """Load a prompt cache from a file.

        :param prompt_cache_path: File path to the prompt cache.
        :raises RuntimeError: If loading fails.
        """
        rc = lib.rkllm_load_prompt_cache(self._handle[0], prompt_cache_path)
        if rc != 0:
            raise RuntimeError(f"Failed to load prompt cache, error code: {rc}")

    def release_prompt_cache(self):
        """Release the prompt cache from memory.

        :raises RuntimeError: If releasing fails.
        """
        rc = lib.rkllm_release_prompt_cache(self._handle[0])
        if rc != 0:
            raise RuntimeError(f"Failed to release prompt cache, error code: {rc}")

    @overload  # RKLLMInputType.MULTIMODAL
    def run(
        self, *, prompt: bytes, image_embed: tuple[float], n_image_tokens: int, **kwargs
    ) -> int: ...

    @overload  # RKLLMInputType.TOKEN
    def run(self, *, input_ids: tuple[int], n_tokens: int, **kwargs) -> int: ...

    @overload  # RKLLMInputType.EMBED
    def run(self, *, embed: tuple[float], n_tokens: int, **kwargs) -> int: ...

    @overload  # RKLLMInputType.PROMPT
    def run(self, *, prompt: bytes, **kwargs) -> int: ...

    def run(self, **kwargs) -> int:
        """Run an inference task synchronously.

        :return: The status code (0 indicates success).
        """
        match kwargs:
            case {
                "prompt": prompt,
                "image_embed": image_embed,
                "n_image_tokens": n_image_tokens,
                **kw,
            }:
                input_args = {
                    "input_type": RKLLMInputType.MULTIMODAL,
                    "multimodal_input": RKLLMMultiModelInput(
                        prompt=prompt,
                        image_embed=image_embed,
                        n_image_tokens=n_image_tokens,
                    )[0],  # type: ignore[index]
                }

            case {"input_ids": input_ids, "n_tokens": n_tokens, **kw}:
                input_args = {
                    "input_type": RKLLMInputType.TOKEN,
                    "token_input": RKLLMTokenInput(
                        input_ids=input_ids,
                        n_tokens=n_tokens,
                    )[0],  # type: ignore[index]
                }

            case {"embed": embed, "n_tokens": n_tokens, **kw}:
                input_args = {
                    "input_type": RKLLMInputType.EMBED,
                    "embed_input": RKLLMEmbedInput(
                        embed=embed,
                        n_tokens=n_tokens,
                    )[0],  # type: ignore[index]
                }

            case {"prompt": prompt, **kw}:
                if "image_embed" in kw:
                    raise ValueError("Missing keyword argument: 'n_image_tokens'")
                if "n_image_tokens" in kw:
                    raise ValueError("Missing keyword argument: 'image_embed'")

                input_args = {
                    "input_type": RKLLMInputType.PROMPT,
                    "prompt_input": ffi.new("char[]", prompt),
                }

            case _:
                raise ValueError("Invalid input arguments")

        infer_input = ffi.new("RKLLMInput *", input_args)
        infer_params = RKLLMInferParam(  # type: ignore[call-arg]
            mode=RKLLMInferMode.GENERATE,
            # lora_params=None,  # NotImplemented
            # prompt_cache_params=None,  # NotImplemented
        )

        return lib.rkllm_run(self._handle[0], infer_input, infer_params, ffi.NULL)

    def abort(self):
        """Abort an ongoing inference task.

        :raises RuntimeError: If aborting fails.
        """
        rc = lib.rkllm_abort(self._handle[0])
        if rc != 0:
            raise RuntimeError(f"Failed to abort LLM task, error code: {rc}")

    def is_running(self) -> bool:
        """Check whether an inference task is currently running.

        :return: True if a task is running, False otherwise.
        """
        rc = lib.rkllm_is_running(self._handle[0])
        return rc == 0

    def destroy(self):
        """Destroy the LLM instance and release resources.

        :raises RuntimeError: If destruction fails.
        """
        if self._handle:
            rc = lib.rkllm_destroy(self._handle[0])
            if rc != 0:
                raise RuntimeError(f"LLM destroy failed with error code {rc}")
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    def __del__(self):
        if self._handle:
            self.destroy()
