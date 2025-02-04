from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_community.llms import HuggingFaceTextGenInference
from langchain_core.outputs import LLMResult
from pydantic import Field
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs import GenerationChunk
from langchain_dartmouth.definitions import LLM_BASE_URL, CLOUD_BASE_URL
from langchain_dartmouth.base import AuthenticatedMixin
from langchain_dartmouth.model_listing import (
    DartmouthModelListing,
    CloudModelListing,
    reformat_model_spec,
)

from openai import DefaultHttpxClient, DefaultAsyncHttpxClient

import os

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
)


class DartmouthLLM(HuggingFaceTextGenInference, AuthenticatedMixin):
    r"""
    Dartmouth-deployed Large Language Models. Use this class for non-chat models
    (e.g., `CodeLlama 13B <https://huggingface.co/codellama/CodeLlama-13b-Python-hf>`_).

    This class does **not** format the prompt to adhere to any required templates.
    The string you pass to it is exactly the string received by the LLM. If the
    desired model requires a chat template (e.g.,
    `Llama 3.1 Instruct <https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/>`_),
    you may want to use :class:`ChatDartmouth` instead.

    :param model_name: Name of the model to use, defaults to ``"codellama-13b-python-hf"``.
    :type model_name: str, optional
    :param temperature: Temperature to use for sampling (higher temperature means more varied outputs), defaults to ``0.8``.
    :type temperature: float, optional
    :param max_new_tokens: Maximum number of generated tokens, defaults to ``512``.
    :type max_new_tokens: int
    :param streaming: Whether to generate a stream of tokens asynchronously, defaults to ``False``
    :type streaming: bool
    :param top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    :type top_k: int, optional
    :param top_p: If set to < 1, only the smallest set of most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation, defaults to ``0.95``.
    :type top_p: float, optional
    :param typical_p: Typical Decoding mass. See `Typical Decoding for Natural Language Generation <https://arxiv.org/abs/2202.00666>`_ for more information, defaults to ``0.95``.
    :type typical_p: float, optional
    :param repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`_ for more details.
    :type repetition_penalty: float, optional
    :param return_full_text: Whether to prepend the prompt to the generated text, defaults to ``False``
    :type return_full_text: bool
    :param truncate: Truncate inputs tokens to the given size
    :type truncate: int, optional
    :param stop_sequences: Stop generating tokens if a member of ``stop_sequences`` is generated.
    :type stop_sequences: List[str], optional
    :param seed: Random sampling seed
    :type seed: int, optional
    :param do_sample: Activate logits sampling, defaults to ``False``.
    :type do_sample: bool
    :param watermark: Watermarking with `A Watermark for Large Language Models <https://arxiv.org/abs/2301.10226>`_, defaults to ``False``
    :type watermark: bool
    :param model_kwargs: Parameters to pass to the model (see the documentation of LangChain's `HuggingFaceTextGenInference class <https://api.python.langchain.com/en/latest/llms/langchain_community.llms.huggingface_text_gen_inference.HuggingFaceTextGenInference.html>`_.)
    :type model_kwargs: dict, optional
    :param dartmouth_api_key: A Dartmouth API key (obtainable from https://developer.dartmouth.edu). If not specified, it is attempted to be inferred from an environment variable DARTMOUTH_API_KEY.
    :type dartmouth_api_key: str, optional
    :param authenticator: A Callable returning a JSON Web Token (JWT) for authentication.
    :type authenticator: Callable, optional
    :param jwt_url: URL of the Dartmouth API endpoint returning a JSON Web Token (JWT).
    :type jwt_url: str, optional
    :param inference_server_url: URL pointing to an inference endpoint, defaults to ``"https://ai-api.dartmouth.edu/tgi/"``.
    :type inference_server_url: str
    :param timeout: Timeout in seconds, defaults to ``120``
    :type timeout: int
    :param server_kwargs: Holds any text-generation-inference server parameters not explicitly specified
    :type server_kwargs: dict, optional
    :param \**_: Additional keyword arguments are silently discarded. This is to ensure interface compatibility with other langchain components.

    Example
    --------

    With an environment variable named ``DARTMOUTH_API_KEY`` pointing to your key obtained from `https://developer.dartmouth.edu <https://developer.dartmouth.edu>`_, using a Dartmouth-hosted LLM only takes a few lines of code:

    .. code-block:: python

        from langchain_dartmouth.llms import DartmouthLLM

        llm = DartmouthLLM(model_name="codellama-13b-hf")

        response = llm.invoke("Write a Python script to swap two variables."")
        print(response)
    """

    authenticator: Optional[Callable] = None
    dartmouth_api_key: Optional[str] = None
    jwt_url: Optional[str] = None
    max_new_tokens: int = 512
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    typical_p: Optional[float] = 0.95
    temperature: Optional[float] = 0.8
    repetition_penalty: Optional[float] = None
    return_full_text: bool = False
    truncate: Optional[int] = None
    stop_sequences: List[str] = Field(default_factory=list)
    seed: Optional[int] = None
    inference_server_url: str = ""
    timeout: int = 120
    streaming: bool = False
    do_sample: bool = False
    watermark: bool = False
    server_kwargs: Dict[str, Any] = Field(default_factory=dict)
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        model_name: str = "codellama-13b-python-hf",
        temperature: float = 0.8,
        max_new_tokens: int = 512,
        streaming: bool = False,
        top_k: int = None,
        top_p: float = None,
        typical_p: float = None,
        repetition_penalty: float = None,
        return_full_text: bool = False,
        truncate: int = None,
        stop_sequences: List[str] = None,
        seed: int = None,
        do_sample: bool = False,
        watermark: bool = False,
        model_kwargs: Optional[dict] = None,
        dartmouth_api_key: Optional[str] = None,
        authenticator: Optional[Callable] = None,
        jwt_url: Optional[str] = None,
        inference_server_url: Optional[str] = "",
        timeout: int = 120,
        server_kwargs: Dict[str, Any] = None,
        **_,
    ):
        """Initializes the object"""
        if not inference_server_url:
            inference_server_url = f"{LLM_BASE_URL}{model_name}/"
        # Explicitly pass kwargs to control which ones show up in the documentation
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "top_k": top_k,
            "top_p": top_p,
            "typical_p": typical_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "return_full_text": return_full_text,
            "truncate": truncate,
            "stop_sequences": stop_sequences,
            "seed": seed,
            "timeout": timeout,
            "streaming": streaming,
            "do_sample": do_sample,
            "watermark": watermark,
            "server_kwargs": server_kwargs if server_kwargs is not None else {},
            "inference_server_url": inference_server_url,
            "model_kwargs": model_kwargs if model_kwargs is not None else {},
        }
        super().__init__(**kwargs)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.jwt_url = jwt_url
        self.authenticate(jwt_url=self.jwt_url)

    @staticmethod
    def list(dartmouth_api_key: str = None) -> list[dict]:
        """List the models available through ``DartmouthLLM``.

        :param dartmouth_api_key: A Dartmouth API key (obtainable from https://developer.dartmouth.edu). If not specified, it is attempted to be inferred from an environment variable ``DARTMOUTH_API_KEY``.
        :type dartmouth_api_key: str, optional
        :return: A list of descriptions of the available models
        :rtype: list[dict]
        """
        try:
            if dartmouth_api_key is None:
                dartmouth_api_key = os.environ["DARTMOUTH_API_KEY"]
        except KeyError as e:
            raise KeyError(
                "Dartmouth API key not provided as argument or defined as environment variable 'DARTMOUTH_API_KEY'."
            ) from e
        listing = DartmouthModelListing(api_key=dartmouth_api_key)
        models = listing.list(server="text-generation-inference", type="llm")
        return models

    def invoke(self, *args, **kwargs) -> str:
        """Transforms a single input into an output.

        See `LangChain's API documentation <https://python.langchain.com/v0.1/docs/expression_language/interface/>`_ for details on how to use this method.

        :return: The LLM's completion of the input string.
        :rtype: str
        """
        try:
            return super().invoke(*args, **kwargs)
        except KeyError:
            self.authenticate(jwt_url=self.jwt_url)
            return super().invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs) -> str:
        """Asynchronously transforms a single input into an output.

        See `LangChain's API documentation <https://python.langchain.com/v0.1/docs/expression_language/interface/>`_ for details on how to use this method.

        :return: The LLM's completion of the input string.
        :rtype: str
        """
        try:
            return super().ainvoke(*args, **kwargs)
        except KeyError:
            self.authenticate(jwt_url=self.jwt_url)
            return super().ainvoke(*args, **kwargs)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        try:
            for chunk in super()._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            for chunk in super()._stream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        try:
            async for chunk in super()._astream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk

        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            async for chunk in super()._astream(
                prompt=prompt, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk


def DartmouthChatModel(*args, **kwargs):
    from warnings import warn

    warn(
        "DartmouthChatModel is deprecated and will be removed in a future update. Use `DartmouthLLM` (as a drop-in replacement) or `ChatDartmouth` instead!"
    )
    return DartmouthLLM(*args, **kwargs)


class ChatDartmouth(ChatOpenAI, AuthenticatedMixin):
    r"""Dartmouth-deployed Chat models (also known as Instruct models).

    Use this class if you want to use a model that uses a chat template
    (e.g., `Llama 3.1 8B Instruct <https://huggingface.co/meta-llama/meta-llama-3.1-8b-instruct>`_).

    All prompts are automatically formatted to adhere to the chosen model's
    chat template. If you need more control over the exact string sent to the model,
    you may want to use :class:`DartmouthLLM` instead.

    :param model_name: Name of the model to use, defaults to ``"llama-3-1-8b-instruct"``.
    :type model_name: str
    :param streaming: Whether to stream the results or not, defaults to ``False``.
    :type streaming: bool
    :param temperature: Temperature to use for sampling (higher temperature means more varied outputs), defaults to ``0.7``.
    :type temperature: float
    :param max_tokens: Maximum number of tokens to generate, defaults to 512
    :type max_tokens: int
    :param logprobs: Whether to return logprobs
    :type logprobs: bool, optional
    :param stream_usage: Whether to include usage metadata in streaming output. If ``True``, additional message chunks will be generated during the stream including usage metadata, defaults to ``False``.
    :type stream_usage: bool
    :param presence_penalty: Penalizes repeated tokens.
    :type presence_penalty: float, optional
    :param frequency_penalty: Penalizes repeated tokens according to frequency.
    :type frequency_penalty: float, optional
    :param seed: Seed for generation
    :type seed: int, optional
    :param top_logprobs: Number of most likely tokens to return at each token position, each with an associated log probability. ``logprobs`` must be set to true if this parameter is used.
    :type top_logprobs: int, optional
    :param logit_bias: Modify the likelihood of specified tokens appearing in the completion.
    :type logit_bias: dict, optional
    :param n: Number of chat completions to generate for each prompt, defaults to ``1``
    :type n: int
    :param top_p: Total probability mass of tokens to consider at each step.
    :type top_p: float, optional
    :param model_kwargs: Holds any model parameters valid for ``create`` call not explicitly specified.
    :type model_kwargs: dict, optional
    :param dartmouth_api_key: A Dartmouth API key (obtainable from https://developer.dartmouth.edu). If not specified, it is attempted to be inferred from an environment variable DARTMOUTH_API_KEY.
    :type dartmouth_api_key: str, optional
    :param authenticator: A Callable returning a JSON Web Token (JWT) for authentication.
    :type authenticator: Callable, optional
    :param jwt_url: URL of the Dartmouth API endpoint returning a JSON Web Token (JWT).
    :type jwt_url: str, optional
    :param inference_server_url: URL pointing to an inference endpoint, defaults to ``"https://ai-api.dartmouth.edu/tgi/"``.
    :type inference_server_url: str, optional
    :param \**_: Additional keyword arguments are silently discarded. This is to ensure interface compatibility with other langchain components.


    Example
    ----------

    With an environment variable named ``DARTMOUTH_API_KEY`` pointing to your key obtained from `https://developer.dartmouth.edu <https://developer.dartmouth.edu>`_, using a Dartmouth-hosted LLM only takes a few lines of code:

    .. code-block:: python

        from langchain_dartmouth.llms import ChatDartmouth

        llm = ChatDartmouth(model_name="llama-3-8b-instruct")

        response = llm.invoke("Hi there!")

        print(response.content)

    .. note::

        The required prompt format is enforced automatically when you are using ``ChatDartmouth``.
    """

    authenticator: Optional[Callable] = None
    dartmouth_api_key: Optional[str] = None
    jwt_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    stream_usage: bool = False
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    logit_bias: Optional[Dict[int, int]] = None
    streaming: bool = False
    n: int = 1
    top_p: Optional[float] = None
    model_kwargs: Optional[dict] = None
    model_name: str = Field(default="llama-3-1-8b-instruct")

    def __init__(
        self,
        model_name: str = "llama-3-1-8b-instruct",
        streaming: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 512,
        logprobs: Optional[bool] = None,
        stream_usage: bool = False,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[int, int]] = None,
        n: int = 1,
        top_p: Optional[float] = None,
        model_kwargs: Optional[dict] = None,
        dartmouth_api_key: Optional[str] = None,
        authenticator: Optional[Callable] = None,
        jwt_url: Optional[str] = None,
        inference_server_url: Optional[str] = None,
        **_,
    ):
        # Explicitly pass kwargs to control which ones show up in the documentation
        kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream_usage": stream_usage,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "logit_bias": logit_bias,
            "streaming": streaming,
            "n": n,
            "top_p": top_p,
            "model_kwargs": model_kwargs if model_kwargs is not None else {},
        }
        if inference_server_url:
            kwargs["openai_api_base"] = inference_server_url
        else:
            kwargs["openai_api_base"] = f"{LLM_BASE_URL}{model_name}/v1/"
        # For compliance, a non-null API key must be set
        kwargs["openai_api_key"] = "unused"
        super().__init__(**kwargs)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.jwt_url = jwt_url
        self.authenticate(jwt_url=self.jwt_url)

    @staticmethod
    def list(dartmouth_api_key: str = None) -> list[dict]:
        """List the models available through ``ChatDartmouth``.

        :param dartmouth_api_key: A Dartmouth API key (obtainable from https://developer.dartmouth.edu). If not specified, it is attempted to be inferred from an environment variable ``DARTMOUTH_API_KEY``.
        :type dartmouth_api_key: str, optional
        :return: A list of descriptions of the available models
        :rtype: list[dict]
        """
        try:
            if dartmouth_api_key is None:
                dartmouth_api_key = os.environ["DARTMOUTH_API_KEY"]
        except KeyError as e:
            raise KeyError(
                "Dartmouth API key not provided as argument or defined as environment variable 'DARTMOUTH_API_KEY'."
            ) from e
        listing = DartmouthModelListing(api_key=dartmouth_api_key)
        models = listing.list(
            server="text-generation-inference", type="llm", capabilities=["chat"]
        )
        return models

    def invoke(self, *args, **kwargs) -> BaseMessage:
        """Invokes the model to get a response to a query.

        See `LangChain's API documentation <https://python.langchain.com/v0.1/docs/expression_language/interface/>`_ for details on how to use this method.

        :return: The LLM's response to the prompt.
        :rtype: BaseMessage
        """
        try:
            return super().invoke(*args, **kwargs)
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            return super().invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs) -> BaseMessage:
        """Asynchronously invokes the model to get a response to a query.

        See `LangChain's API documentation <https://python.langchain.com/v0.1/docs/expression_language/interface/>`_ for details on how to use this method.

        :return: The LLM's response to the prompt.
        :rtype: BaseMessage
        """
        try:
            response = await super().ainvoke(*args, **kwargs)
            return response
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            response = await super().ainvoke(*args, **kwargs)
            return response

    def stream(self, *args, **kwargs) -> Iterator[BaseMessageChunk]:
        try:
            for chunk in super().stream(*args, **kwargs):
                yield chunk
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            for chunk in super().stream(*args, **kwargs):
                yield chunk

    async def astream(self, *args, **kwargs) -> AsyncIterator[BaseMessageChunk]:
        try:
            async for chunk in super().astream(*args, **kwargs):
                yield chunk
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            async for chunk in super().astream(*args, **kwargs):
                yield chunk

    def generate(self, *args, **kwargs) -> LLMResult:
        try:
            return super().generate(*args, **kwargs)
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            return super().generate(*args, **kwargs)

    async def agenerate(self, *args, **kwargs) -> LLMResult:
        try:
            response = await super().agenerate(*args, **kwargs)
            return response
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            response = await super().agenerate(*args, **kwargs)
            return response


class ChatDartmouthCloud(ChatOpenAI):
    r"""Cloud chat models made available by Dartmouth.

    Use this class if you want to use a model by a third-party provider, e.g., Anthropic or OpenAI, made accessible by Dartmouth.

    :param model_name: Name of the model to use, defaults to ``"openai.gpt-4o-mini-2024-07-18"``.
    :type model_name: str
    :param streaming: Whether to stream the results or not, defaults to ``False``.
    :type streaming: bool
    :param temperature: Temperature to use for sampling (higher temperature means more varied outputs), defaults to ``0.7``.
    :type temperature: float
    :param max_tokens: Maximum number of tokens to generate, defaults to 512
    :type max_tokens: int
    :param logprobs: Whether to return logprobs
    :type logprobs: bool, optional
    :param stream_usage: Whether to include usage metadata in streaming output. If ``True``, additional message chunks will be generated during the stream including usage metadata, defaults to ``False``.
    :type stream_usage: bool
    :param presence_penalty: Penalizes repeated tokens.
    :type presence_penalty: float, optional
    :param frequency_penalty: Penalizes repeated tokens according to frequency.
    :type frequency_penalty: float, optional
    :param seed: Seed for generation
    :type seed: int, optional
    :param top_logprobs: Number of most likely tokens to return at each token position, each with an associated log probability. ``logprobs`` must be set to true if this parameter is used.
    :type top_logprobs: int, optional
    :param logit_bias: Modify the likelihood of specified tokens appearing in the completion.
    :type logit_bias: dict, optional
    :param n: Number of chat completions to generate for each prompt, defaults to ``1``
    :type n: int
    :param top_p: Total probability mass of tokens to consider at each step.
    :type top_p: float, optional
    :param model_kwargs: Holds any model parameters valid for ``create`` call not explicitly specified.
    :type model_kwargs: dict, optional
    :param dartmouth_chat_api_key: A Dartmouth Chat API key (see `here <https://rcweb.dartmouth.edu/~d20964h/2024-12-11-dartmouth-chat-api/api_key/>`_ for how to obtain one). If not specified, it is attempted to be inferred from an environment variable ``DARTMOUTH_CHAT_API_KEY``.
    :type dartmouth_chat_api_key: str, optional
    :param \**_: Additional keyword arguments are silently discarded. This is to ensure interface compatibility with other langchain components.


    Example
    ----------

    With an environment variable named ``DARTMOUTH_CHAT_API_KEY`` pointing to your key obtained from `https://chat.dartmouth.edu <https://chat.dartmouth.edu>`_, using a third-party LLM provided by Dartmouth only takes a few lines of code:

    .. code-block:: python

        from langchain_dartmouth.llms import ChatDartmouthCloud

        llm = ChatDartmouthCloud(model_name="openai.gpt-4o-mini-2024-07-18")

        response = llm.invoke("Hi there!")

        print(response.content)

    .. note::

        The models available through ``ChatDartmouthCloud`` are pay-as-you-go third-party models. Dartmouth pays for the use, but a daily token limit per user applies.
    """

    dartmouth_chat_api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    stream_usage: bool = False
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    seed: Optional[int] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    logit_bias: Optional[Dict[int, int]] = None
    streaming: bool = False
    n: int = 1
    top_p: Optional[float] = None
    model_kwargs: Optional[dict] = None
    model_name: str = Field(default="openai.gpt-4o-mini-2024-07-18")

    def __init__(
        self,
        model_name: str = "openai.gpt-4o-mini-2024-07-18",
        streaming: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 512,
        logprobs: Optional[bool] = None,
        stream_usage: bool = False,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        logit_bias: Optional[Dict[int, int]] = None,
        n: int = 1,
        top_p: Optional[float] = None,
        model_kwargs: Optional[dict] = None,
        dartmouth_chat_api_key: Optional[str] = None,
        **_,
    ):
        # Explicitly pass kwargs to control which ones show up in the documentation
        kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream_usage": stream_usage,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
            "logit_bias": logit_bias,
            "streaming": streaming,
            "n": n,
            "top_p": top_p,
            "model_kwargs": model_kwargs if model_kwargs is not None else {},
        }
        kwargs["model_name"] = model_name
        kwargs["openai_api_base"] = f"{CLOUD_BASE_URL}"
        if dartmouth_chat_api_key is None:
            try:
                dartmouth_chat_api_key = os.environ["DARTMOUTH_CHAT_API_KEY"]
            except KeyError as e:
                raise KeyError(
                    "Dartmouth Chat API key not provided as argument or defined as environment variable 'DARTMOUTH_CHAT_API_KEY'."
                ) from e

        kwargs["openai_api_key"] = dartmouth_chat_api_key
        super().__init__(
            **kwargs,
            # Turn off following redirects, see issue #8
            http_async_client=DefaultAsyncHttpxClient(follow_redirects=False),
            http_client=DefaultHttpxClient(follow_redirects=False),
        )
        self.dartmouth_chat_api_key = dartmouth_chat_api_key

    @staticmethod
    def list(dartmouth_chat_api_key: str = None) -> list[dict]:
        """List the models available through ``ChatDartmouthCloud``.

        :param dartmouth_chat_api_key: A Dartmouth Chat API key (obtainable from `https://chat.dartmouth.edu <https://chat.dartmouth.edu>`_). If not specified, it is attempted to be inferred from an environment variable ``DARTMOUTH_CHAT_API_KEY``.
        :type dartmouth_chat_api_key: str, optional
        :return: A list of descriptions of the available models
        :rtype: list[dict]
        """
        try:
            if dartmouth_chat_api_key is None:
                dartmouth_chat_api_key = os.environ["DARTMOUTH_CHAT_API_KEY"]
        except KeyError as e:
            raise KeyError(
                "Dartmouth Chat API key not provided as argument or defined as environment variable 'DARTMOUTH_CHAT_API_KEY'."
            ) from e
        listing = CloudModelListing(api_key=dartmouth_chat_api_key)
        models = listing.list()

        # Filter out some models better accessed through other means
        def is_cloud_model(m):
            return not (
                m["id"].startswith("meta.")
                or m["id"].startswith("ollama.")
                or m["owned_by"] in ["ollama", "arena"]
            )

        models = [reformat_model_spec(m) for m in models if is_cloud_model(m)]
        return models

    def invoke(self, *args, **kwargs) -> BaseMessage:
        """Invokes the model to get a response to a query.

        See `LangChain's API documentation <https://python.langchain.com/v0.1/docs/expression_language/interface/>`_ for details on how to use this method.

        :return: The LLM's response to the prompt.
        :rtype: BaseMessage
        """
        return super().invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs) -> BaseMessage:
        """Asynchronously invokes the model to get a response to a query.

        See `LangChain's API documentation <https://python.langchain.com/v0.1/docs/expression_language/interface/>`_ for details on how to use this method.

        :return: The LLM's response to the prompt.
        :rtype: BaseMessage
        """
        response = await super().ainvoke(*args, **kwargs)
        return response

    def stream(self, *args, **kwargs) -> Iterator[BaseMessageChunk]:
        for chunk in super().stream(*args, **kwargs):
            yield chunk

    async def astream(self, *args, **kwargs) -> AsyncIterator[BaseMessageChunk]:
        async for chunk in super().astream(*args, **kwargs):
            yield chunk

    def generate(self, *args, **kwargs) -> LLMResult:
        return super().generate(*args, **kwargs)

    async def agenerate(self, *args, **kwargs) -> LLMResult:
        response = await super().agenerate(*args, **kwargs)
        return response


if __name__ == "__main__":
    print(DartmouthLLM.list())
    print(ChatDartmouth.list())
    print(ChatDartmouthCloud.list())
