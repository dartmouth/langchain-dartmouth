from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

import os
from typing import Callable, List, Optional

from langchain_dartmouth.base import AuthenticatedMixin
from langchain_dartmouth.definitions import EMBEDDINGS_BASE_URL
from langchain_dartmouth.model_listing import DartmouthModelListing


class DartmouthEmbeddings(HuggingFaceEndpointEmbeddings, AuthenticatedMixin):
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

        from langchain_dartmouth.embeddings import DartmouthEmbeddingsModel


        embeddings = DartmouthEmbeddingsModel()

        response = embeddings.embed_query("Hello? Is there anybody in there?")

        print(response)
    """

    authenticator: Optional[Callable] = None
    dartmouth_api_key: Optional[str] = None
    jwt_url: Optional[str] = None
    embeddings_server_url: Optional[str] = None

    def __init__(
        self,
        model_name: str = "bge-large-en-v1-5",
        model_kwargs: Optional[dict] = None,
        dartmouth_api_key: Optional[str] = None,
        authenticator: Optional[Callable] = None,
        jwt_url: Optional[str] = None,
        embeddings_server_url: Optional[str] = None,
    ):
        """Initializes the object"""
        if embeddings_server_url:
            endpoint = f"{embeddings_server_url}{model_name}/"
        else:
            endpoint = f"{EMBEDDINGS_BASE_URL}{model_name}/"
        super().__init__(model=endpoint, model_kwargs=model_kwargs)
        self.authenticator = authenticator
        self.dartmouth_api_key = dartmouth_api_key
        self.authenticate(jwt_url=jwt_url)

    @staticmethod
    def list(dartmouth_api_key: str = None) -> list[dict]:
        """List the models available through ``DartmouthEmbeddings``.

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
        models = listing.list(server="text-embeddings-inference", type="embedding")
        return models

    def embed_query(self, text: str) -> List[float]:
        """Call out to the embedding endpoint to retrieve the embedding of the query text.

        :param text: The text to embed.
        :type text: str
        :return: Embeddings for the text.
        :rtype: List[float]
        """
        try:
            return super().embed_query(text)
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            return super().embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to the embedding endpoint to retrieve the embeddings of multiple texts.

        :param text: The list of texts to embed.
        :type text: str
        :return: Embeddings for the texts.
        :rtype: List[List[float]]
        """
        try:
            return super().embed_documents(texts)
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            return super().embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async Call to the embedding endpoint to retrieve the embedding of the query text.

        :param text: The text to embed.
        :type text: str
        :return: Embeddings for the text.
        :rtype: List[float]
        """
        try:
            response = await super().aembed_query(text)
            return response
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            response = await super().aembed_query(text)
            return response

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async Call to the embedding endpoint to retrieve the embeddings of multiple texts.

        :param text: The list of texts to embed.
        :type text: str
        :return: Embeddings for the texts.
        :rtype: List[List[float]]
        """
        try:
            response = await super().aembed_documents(texts)
            return response
        except Exception:
            self.authenticate(jwt_url=self.jwt_url)
            response = await super().aembed_documents(texts)
            return response

if __name__ == "__main__":
    print(DartmouthEmbeddings.list())
