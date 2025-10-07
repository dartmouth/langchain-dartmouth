from langchain_dartmouth.definitions import USER_AGENT
from langchain_openai import OpenAIEmbeddings
from openai import DefaultHttpxClient, DefaultAsyncHttpxClient
import os
from typing import Any, Callable, List, Optional

from langchain_dartmouth.definitions import CLOUD_BASE_URL
from langchain_dartmouth.model_listing import CloudModelListing
from langchain_dartmouth.utils import filter_dartmouth_chat_models


class DartmouthEmbeddings(OpenAIEmbeddings):
    """Embedding models deployed on Dartmouth's cluster.

    :param model_name: The name of the embedding model to use, defaults to ``"bge-large-en-v1-5"``.
    :type model_name: str, optional
    :param model_kwargs: Keyword arguments to pass to the model.
    :type model_kwargs: dict, optional
    :param dartmouth_api_key: A Dartmouth API key (obtainable from https://developer.dartmouth.edu). If not specified, it is attempted to be inferred from an environment variable ``DARTMOUTH_API_KEY``.
    :type dartmouth_api_key: str, optional
    :param authenticator: A Callable returning a JSON Web Token (JWT) for authentication. Only needed for special use cases.
    :type authenticator: Callable, optional
    :param jwt_url: URL of the Dartmouth API endpoint returning a JSON Web Token (JWT).
    :type jwt_url: str, optional
    :param embeddings_server_url: URL pointing to an embeddings endpoint, defaults to ``"https://ai-api.dartmouth.edu/tei/"``.
    :type embeddings_server_url: str, optional

    Example
    -----------

    With an environment variable named ``DARTMOUTH_API_KEY`` pointing to your key obtained from `https://developer.dartmouth.edu <https://developer.dartmouth.edu>`_, using a Dartmouth-hosted embedding model only takes a few lines of code:

    .. code-block:: python

        from langchain_dartmouth.embeddings import DartmouthEmbeddings


        embeddings = DartmouthEmbeddings()

        response = embeddings.embed_query("Hello? Is there anybody in there?")

        print(response)
    """

    dartmouth_chat_api_key: Optional[str] = None
    embeddings_server_url: Optional[str] = None

    def __init__(
        self,
        model_name: str = "bge-large-en-v1-5",
        model_kwargs: Optional[dict] = None,
        dartmouth_chat_api_key: Optional[str] = None,
        authenticator: Optional[Callable] = None,
        embeddings_server_base_url: Optional[str] = None,
    ):
        """Initializes the object"""
        kwargs: dict[str, Any] = dict()
        kwargs["default_headers"] = {"User-Agent": USER_AGENT}
        kwargs["model"] = model_name
        if embeddings_server_base_url is not None:
            kwargs["openai_api_base"] = embeddings_server_base_url
        else:
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
    def list(
        dartmouth_chat_api_key: str | None = None,
        base_only: bool = True,
        url: str = CLOUD_BASE_URL,
    ) -> list[dict]:
        """List the models available through ``DartmouthEmbeddings``.

        :param dartmouth_chat_api_key: A Dartmouth Chat API key (obtainable from `https://chat.dartmouth.edu <https://chat.dartmouth.edu>`_). If not specified, it is attempted to be inferred from an environment variable ``DARTMOUTH_CHAT_API_KEY``.
        :type dartmouth_chat_api_key: str, optional
        :param base_only: If True, only regular Large Language Models are returned. If False, Workspace models are also returned.
        :type base_only: bool, optional
        :param url: URL of the listing server
        :type url: str, optional
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
        listing = CloudModelListing(api_key=dartmouth_chat_api_key, url=url)
        models = listing.list(base_only=base_only)
        # Some /models endpoints return the model list in a field:
        if "data" in models:
            models = models["data"]

        return filter_dartmouth_chat_models(models, include_tag="embedding")

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        """Call out to the embedding endpoint to retrieve the embedding of the query text.

        :param text: The text to embed.
        :type text: str
        :return: Embeddings for the text.
        :rtype: List[float]
        """
        return super().embed_query(text, **kwargs)

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = None, **kwargs: Any
    ) -> List[List[float]]:
        """Call out to the embedding endpoint to retrieve the embeddings of multiple texts.

        :param text: The list of texts to embed.
        :type text: str
        :return: Embeddings for the texts.
        :rtype: List[List[float]]
        """
        return super().embed_documents(texts, chunk_size, **kwargs)

    async def aembed_query(self, text: str, **kwargs: Any) -> List[float]:
        """Async Call to the embedding endpoint to retrieve the embedding of the query text.

        :param text: The text to embed.
        :type text: str
        :return: Embeddings for the text.
        :rtype: List[float]
        """
        response = await super().aembed_query(text, **kwargs)
        return response

    async def aembed_documents(
        self, texts: List[str], chunk_size: Optional[int] = None, **kwargs: Any
    ) -> List[List[float]]:
        """Async Call to the embedding endpoint to retrieve the embeddings of multiple texts.

        :param text: The list of texts to embed.
        :type text: str
        :return: Embeddings for the texts.
        :rtype: List[List[float]]
        """
        response = await super().aembed_documents(texts, chunk_size, **kwargs)
        return response


if __name__ == "__main__":
    print([m["name"] for m in DartmouthEmbeddings.list()])
