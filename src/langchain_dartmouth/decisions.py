"""Decision models hosted by Dartmouth.

Decision models answer one or more typed judgments about input text or
structured state -- binary, categorical, or ordinal -- returning probabilities
and confidence instead of free-form text. This module wraps TypeSafe's
``langchain-typesafe`` integration, configured for Dartmouth's gateway and
authentication.
"""

import os
from typing import Optional

import httpx2
from langchain_typesafe import (
    Choice,
    ClassifierRequest,
    ClassifierResponse,
    Noul,
    NoulCriteria,
    Question,
    Score,
    State,
    TypeSafeClassifier,
    Usage,
)
from langchain_typesafe.client import (
    TypeSafeAuthenticationError,
    TypeSafeError,
    TypeSafeNotFoundError,
    TypeSafeRateLimitError,
)

from langchain_dartmouth.definitions import DECISIONS_BASE_URL, USER_AGENT
from langchain_dartmouth.exceptions import (
    DecisionAPIError,
    DecisionRateLimitError,
    InvalidKeyError,
    ModelNotFoundError,
)
from langchain_dartmouth.model_listing import DecisionModelListing, ModelInfo

__all__ = [
    "Choice",
    "ClassifierRequest",
    "ClassifierResponse",
    "DartmouthDecisions",
    "Noul",
    "NoulCriteria",
    "Question",
    "Score",
    "State",
    "Usage",
]


class DartmouthDecisions(TypeSafeClassifier):
    """Decision models made available by Dartmouth.

    Use this class to ask one or more typed questions about text or
    structured state and receive typed judgments with probabilities and
    confidence, rather than a free-form chat completion. A single request can
    combine binary (:class:`~langchain_typesafe.Noul`), categorical
    (:class:`~langchain_typesafe.Choice`), and ordinal
    (:class:`~langchain_typesafe.Score`) questions.

    To see which Decision models are available, run ``DartmouthDecisions.list()``.

    :param model_name: Name of the Decision model to use, defaults to ``"cloudflare.clef"``.
    :type model_name: str, optional
    :param dartmouth_chat_api_key: A Dartmouth Chat API key (see `here <https://rcweb.dartmouth.edu/~d20964h/2024-12-11-dartmouth-chat-api/api_key/>`_ for how to obtain one). If not specified, it is attempted to be inferred from an environment variable ``DARTMOUTH_CHAT_API_KEY``.
    :type dartmouth_chat_api_key: str, optional
    :param inference_server_url: The URL of the inference server (e.g., https://chat.dartmouth.edu/api/)
    :type inference_server_url: str, optional
    :param timeout: Timeout in seconds applied to clients created by this class, defaults to ``30.0``.
    :type timeout: float, optional
    :param client: Optional synchronous ``httpx2.Client`` used by ``invoke``. If omitted, a client is created automatically.
    :type client: httpx2.Client, optional
    :param async_client: Optional asynchronous ``httpx2.AsyncClient`` used by ``ainvoke``. If omitted, a client is created automatically.
    :type async_client: httpx2.AsyncClient, optional

    Example
    ----------

    With an environment variable named ``DARTMOUTH_CHAT_API_KEY`` pointing to your key obtained from `https://chat.dartmouth.edu <https://chat.dartmouth.edu>`_, asking a Decision model to judge some text only takes a few lines of code:

    .. code-block:: python

        from langchain_dartmouth.decisions import DartmouthDecisions, Noul

        decisions = DartmouthDecisions()

        response = decisions.invoke(
            {
                "state": "Production is down. Please help immediately.",
                "questions": {
                    "urgent": Noul(
                        instructions="Does this message require an urgent response?"
                    )
                },
            }
        )
        print(response.nouls["urgent"].noul)
    """

    dartmouth_chat_api_key: Optional[str] = None

    def __init__(
        self,
        model_name: str = "cloudflare.clef",
        dartmouth_chat_api_key: Optional[str] = None,
        inference_server_url: Optional[str] = None,
        timeout: float = 30.0,
        client: Optional[httpx2.Client] = None,
        async_client: Optional[httpx2.AsyncClient] = None,
    ):
        if dartmouth_chat_api_key is None:
            try:
                dartmouth_chat_api_key = os.environ["DARTMOUTH_CHAT_API_KEY"]
            except KeyError as e:
                raise KeyError(
                    "Dartmouth Chat API key not provided as argument or defined as environment variable 'DARTMOUTH_CHAT_API_KEY'."
                ) from e

        # Explicitly pass kwargs to control which ones show up in the documentation
        kwargs = {
            "model": model_name,
            "api_key": dartmouth_chat_api_key,
            "base_url": inference_server_url or DECISIONS_BASE_URL,
            "timeout": timeout,
        }
        if client is not None:
            kwargs["client"] = client
        if async_client is not None:
            kwargs["async_client"] = async_client

        super().__init__(**kwargs)

        self.dartmouth_chat_api_key = dartmouth_chat_api_key

    @staticmethod
    def list(
        dartmouth_chat_api_key: Optional[str] = None,
        url: str = DECISIONS_BASE_URL,
    ) -> list[ModelInfo]:
        """List the Decision models available through ``DartmouthDecisions``.

        :param dartmouth_chat_api_key: A Dartmouth Chat API key (obtainable from `https://chat.dartmouth.edu <https://chat.dartmouth.edu>`_). If not specified, it is attempted to be inferred from an environment variable ``DARTMOUTH_CHAT_API_KEY``.
        :type dartmouth_chat_api_key: str, optional
        :param url: URL of the listing server
        :type url: str, optional
        :return: A list of descriptions of the available models
        :rtype: list[ModelInfo]
        """
        try:
            if dartmouth_chat_api_key is None:
                dartmouth_chat_api_key = os.environ["DARTMOUTH_CHAT_API_KEY"]
        except KeyError as e:
            raise KeyError(
                "Dartmouth Chat API key not provided as argument or defined as environment variable 'DARTMOUTH_CHAT_API_KEY'."
            ) from e
        listing = DecisionModelListing(api_key=dartmouth_chat_api_key, url=url)
        return listing.list()

    @property
    def _request_headers(self) -> dict[str, str]:
        headers = super()._request_headers
        headers["User-Agent"] = USER_AGENT
        return headers

    def invoke(
        self, input: ClassifierRequest, config=None, **kwargs
    ) -> ClassifierResponse:
        """Classify one request synchronously.

        See `LangChain's API documentation <https://python.langchain.com/v0.1/docs/expression_language/interface/>`_ for details on how to use this method.

        :return: Structured Decision model answers and request metadata.
        :rtype: ClassifierResponse
        """
        try:
            return super().invoke(input, config, **kwargs)
        except TypeSafeAuthenticationError as e:
            raise InvalidKeyError("API key invalid!") from e
        except TypeSafeNotFoundError as e:
            raise ModelNotFoundError(
                f"Model {self.model} not found. Please use `DartmouthDecisions.list()` "
                "to verify the model name."
            ) from e
        except TypeSafeRateLimitError as e:
            raise DecisionRateLimitError(
                str(e), retry_after_ms=e.retry_after_ms
            ) from e
        except TypeSafeError as e:
            raise DecisionAPIError(str(e)) from e

    async def ainvoke(
        self, input: ClassifierRequest, config=None, **kwargs
    ) -> ClassifierResponse:
        """Classify one request asynchronously.

        See `LangChain's API documentation <https://python.langchain.com/v0.1/docs/expression_language/interface/>`_ for details on how to use this method.

        :return: Structured Decision model answers and request metadata.
        :rtype: ClassifierResponse
        """
        try:
            return await super().ainvoke(input, config, **kwargs)
        except TypeSafeAuthenticationError as e:
            raise InvalidKeyError("API key invalid!") from e
        except TypeSafeNotFoundError as e:
            raise ModelNotFoundError(
                f"Model {self.model} not found. Please use `DartmouthDecisions.list()` "
                "to verify the model name."
            ) from e
        except TypeSafeRateLimitError as e:
            raise DecisionRateLimitError(
                str(e), retry_after_ms=e.retry_after_ms
            ) from e
        except TypeSafeError as e:
            raise DecisionAPIError(str(e)) from e


if __name__ == "__main__":
    print(DartmouthDecisions.list())
