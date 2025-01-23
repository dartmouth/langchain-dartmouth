"""Helper class to interact with the model listing API"""

from langchain_dartmouth.definitions import MODEL_LISTING_BASE_URL


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
        """Get a list of available models.

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


if __name__ == "__main__":
    import os

    listing = DartmouthModelListing(os.environ["DARTMOUTH_API_KEY"])
    print(listing.list(server="text-generation-inference", capabilities=["chat"]))
