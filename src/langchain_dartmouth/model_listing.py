"""Helper class to interact with the model listing API"""

from langchain_dartmouth.definitions import USER_AGENT

from pydantic import BaseModel, ValidationInfo, model_validator, field_validator
import requests
from dartmouth_auth import get_jwt

from typing import Any, ClassVar, List, Literal


class ModelInfo(BaseModel):
    """A class representing information about a model
    (large language model, embedding model, ...) and
    its capabilities and properties.
    """

    id: str
    """ID to use to access the model"""
    name: str | None = None
    """A human-readable name of the model"""
    description: str | None = None
    """A description of the model (as shown at Dartmouth Chat)"""
    is_embedding: bool | None = None
    """Whether this model can be used as an embedding model"""
    capabilities: list[str] | None = None
    """Capabilities of the model"""
    is_local: bool | None = None
    """
    Whether the model is hosted on-prem by Dartmouth (True),
    or off-prem by a third party (False)."""
    cost: Literal["undefined", "free", "$", "$$", "$$$", "$$$$"] | None = None
    """The relative cost of the model (more '$' signs means more expensive)."""

    _relevant_capabilities: ClassVar[list[str]] = [
        "vision",  # Model can process images
        "usage",  # Model reports token usage in response_metadata
        "reasoning",  # Model supports reasoning_effort
        "hybrid reasoning",  # Model supports reasoning_effort as an optional variable
    ]

    @model_validator(mode="before")
    @classmethod
    def flatten_meta(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "info" in data:
                meta = data["info"].pop("meta")
            elif "meta" in data:
                meta = data.pop("meta")
            else:
                return data
            data["description"] = meta.get("description")
            data["capabilities"] = meta.get("capabilities", {}) | {
                "tags": meta.get("tags", [])
            }

            # Pass all tags, will be validated in field validators
            data["is_local"] = meta.get("tags", [])
            data["is_embedding"] = meta.get("tags", [])
            data["cost"] = meta.get("tags", [])

        return data

    @field_validator("is_embedding", mode="before")
    @classmethod
    def set_is_embedding(cls, v: Any, info: ValidationInfo):
        tags = v or dict()
        tag_names = {t["name"].lower() for t in tags if isinstance(t, dict)}
        return "embedding" in tag_names

    @field_validator("is_local", mode="before")
    @classmethod
    def set_is_local(cls, v: Any, info: ValidationInfo):
        tags = v or dict()
        tag_names = {t["name"].lower() for t in tags if isinstance(t, dict)}
        return "Local".lower() in tag_names

    @field_validator("capabilities", mode="before")
    @classmethod
    def get_capabilities(cls, v: Any, info: ValidationInfo):
        capabilities = [
            c.lower()
            for c, enabled in v.items()
            if enabled and c.lower() in cls._relevant_capabilities
        ] + [
            c["name"].lower()
            for c in v.get("tags", [])
            if c["name"].lower() in cls._relevant_capabilities
        ]

        return capabilities

    @field_validator("cost", mode="before")
    @classmethod
    def extract_cost(cls, v: Any, info: ValidationInfo):
        tags = v
        for tag in tags:
            if tag["name"].lower() == "free":
                return "free"
            if tag["name"].startswith("$"):
                return tag["name"]
        return "undefined"


class BaseModelListing:

    def __init__(self, api_key: str, url: str):
        self.api_key = api_key
        self.SESSION = requests.Session()
        self.SESSION.headers.update({"User-Agent": USER_AGENT})
        self.url = url
        self._authenticate()

    def _authenticate(self):
        """Override this method in the derived class"""
        return NotImplementedError

    def list(self):
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
            resp = self.SESSION.get(url=self.url + "list", params=params)
        except Exception:
            self._authenticate()
            resp = self.SESSION.get(url=self.url + "list")

        resp.raise_for_status()
        return resp.json()["models"]


class CloudModelListing(BaseModelListing):
    def _authenticate(self):
        self.SESSION.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def list(self, base_only: bool = False) -> List[ModelInfo]:
        """Get a list of available Cloud models.

        :param base_only: Whether return only base models or customized models, defaults to False
        :type base_only: bool, optional
        :return: List of model descriptions
        :rtype: List[ModelInfo]
        """
        resp = self.SESSION.get(
            url=self.url + f"v1/models{'/base' if base_only else ''}"
        )
        resp.raise_for_status()
        cloud_models = resp.json()
        if "data" in cloud_models:
            cloud_models = cloud_models["data"]
        return [
            ModelInfo.model_validate(m)
            for m in cloud_models
            if m.get("is_active", True)
        ]


if __name__ == "__main__":
    import os

    models = CloudModelListing(
        api_key=os.environ["DARTMOUTH_CHAT_API_KEY"],
        url="https://chat.dartmouth.edu/api/",
    ).list(base_only=True)

    for model in models:
        print(model)
