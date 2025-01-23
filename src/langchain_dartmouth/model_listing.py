"""Helper class to interact with the model listing API"""

from langchain_dartmouth.definitions import MODEL_LISTING_BASE_URL, CLOUD_BASE_URL


import requests
from dartmouth_auth import get_jwt

from typing import Literal, List


class BaseModelListing:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.SESSION = requests.Session()
        self._authenticate()

    def _authenticate(self):
        """Override this method in the derived class"""
        return NotImplementedError

    def list():
        """Override this method in the derived class"""
        return NotImplementedError


class DartmouthModelListing(BaseModelListing):

    def _authenticate(self):
        self.SESSION.headers.update(
            {"Authorization": f"Bearer {get_jwt(dartmouth_api_key=self.api_key)}"}
        )

    def list(self, **kwargs) -> List[dict]:
        """Get a list of available on-premise models.

        Optionally filter by various parameters.

        :return: List of model descriptions
        :rtype: List[dict]
        """
        params = {}
        if "server" in kwargs:
            params["server"] = kwargs["server"]
        if "type" in kwargs:
            params["model_type"] = kwargs["type"]
        if "capabilities" in kwargs:
            params["capability"] = kwargs["capabilities"]

        try:
            resp = self.SESSION.get(url=MODEL_LISTING_BASE_URL + "list", params=params)
        except Exception:
            self._authenticate()
            resp = self.SESSION.get(url=MODEL_LISTING_BASE_URL + "list")

        resp.raise_for_status()
        return resp.json()["models"]


class CloudModelListing(BaseModelListing):
    def _authenticate(self):
        self.SESSION.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def list(self, base_only: bool = False) -> List[dict]:
        """Get a list of available Cloud models.

        :param base_only: Whether return only base models or customized models, defaults to False
        :type base_only: bool, optional
        :return: List of model descriptions
        :rtype: List[dict]
        """
        resp = self.SESSION.get(
            url=CLOUD_BASE_URL + f"models{'/base' if base_only else ''}"
        )
        resp.raise_for_status()
        return resp.json()["data"]


def reformat_model_spec(model_spec: dict) -> dict:
    """Reformats the model specification returned by Open WebUI to align with our on-premise spec format.

    :param model_spec: Model spec returned from Open WebUI
    :type model_spec: dict
    :return: Model spec using the schema of Dartmouth's listing API.
    :rtype: dict
    """
    new_spec = dict()
    new_spec["name"] = model_spec["id"]
    new_spec["provider"] = model_spec["id"].split(sep=".", maxsplit=1)[0]
    new_spec["type"] = "llm"

    def get_capablities(caps: dict) -> List[str]:
        capabilities = ["chat"]  # All models have chat capability
        if caps["vision"]:
            capabilities.append("vision")
        return capabilities

    new_spec["capabilities"] = get_capablities(
        model_spec["info"]["meta"]["capabilities"]
    )
    new_spec["server"] = "dartmouth-chat"
    new_spec["parameters"] = dict()
    return new_spec


if __name__ == "__main__":
    import os

    listing = DartmouthModelListing(os.environ["DARTMOUTH_API_KEY"])
    print(listing.list(server="text-generation-inference", capabilities=["chat"]))

    listing = CloudModelListing(os.environ["DARTMOUTH_CHAT_API_KEY"])
    print(listing.list())
