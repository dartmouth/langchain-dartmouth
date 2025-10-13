"""Model Listing and Information Classes.

This module provides classes and utilities for discovering and querying available
AI models through Dartmouth's infrastructure. It includes functionality for listing
models from both on-premises and cloud-based services, along with detailed metadata
about each model's capabilities, costs, and properties.

The primary classes in this module are:

- :class:`ModelInfo`: A Pydantic model representing detailed information about a
  specific AI model, including its capabilities, hosting location, and cost.
- :class:`DartmouthModelListing`: Interface for listing on-premises models hosted
  by Dartmouth.
- :class:`CloudModelListing`: Interface for listing cloud-based models available
  through Dartmouth Chat.

Model listing functions are integrated into the respective model class interfaces.
For example, you can call :meth:`ChatDartmouth.list()` or
:meth:`DartmouthEmbeddings.list()` to discover available models for those specific
use cases.

Examples
--------
Listing available cloud models:

>>> from langchain_dartmouth.model_listing import CloudModelListing
>>> import os
>>> listing = CloudModelListing(
...     api_key=os.environ["DARTMOUTH_CHAT_API_KEY"],
...     url=os.environ["LCD_CLOUD_BASE_URL"]
... )
>>> models = listing.list(base_only=True)
>>> for model in models:
...     print(f"{model.name}: {model.description}")

Accessing model information:

>>> model = models[0]
>>> print(f"Model ID: {model.id}")
>>> print(f"Capabilities: {model.capabilities}")
>>> print(f"Cost: {model.cost}")
>>> print(f"Local hosting: {model.is_local}")

Notes
-----
The model listing functionality requires appropriate API credentials and access
to Dartmouth's AI infrastructure. Model availability and metadata may vary based
on your access level and the current deployment configuration.
"""

from langchain_dartmouth.definitions import USER_AGENT

from pydantic import BaseModel, ValidationInfo, model_validator, field_validator
import requests
from dartmouth_auth import get_jwt

from typing import Any, ClassVar, List, Literal


class ModelInfo(BaseModel):
    """A class representing information about a model.

    This class encapsulates metadata about language models, embedding models,
    and other AI models available through Dartmouth's infrastructure. It provides
    a structured way to access model properties, capabilities, and configuration
    details.

    The class automatically processes and validates model metadata from API responses,
    extracting relevant information from nested structures and tags.

    Attributes
    ----------
    id : str
        Unique identifier used to access the model in API calls.
    name : str | None
        Human-readable name of the model for display purposes.
    description : str | None
        Detailed description of the model's purpose and characteristics,
        as displayed in Dartmouth Chat interface.
    is_embedding : bool | None
        Flag indicating whether this model can be used for generating embeddings.
    capabilities : list[str] | None
        List of model capabilities such as 'vision', 'tool calling', 'reasoning', etc.
    is_local : bool | None
        Indicates model hosting location:
        - True: Model is hosted on-premises by Dartmouth
        - False: Model is hosted off-premises by a third-party provider
    cost : Literal["undefined", "free", "$", "$$", "$$$", "$$$$"] | None
        Relative cost indicator for model usage:
        - "free": No cost for usage
        - "$" to "$$$$": Increasing cost levels
        - "undefined": Cost information not available


    Examples
    --------
    >>> model = ModelInfo(
    ...     id="gpt-4",
    ...     name="GPT-4",
    ...     description="Advanced language model",
    ...     is_embedding=False,
    ...     capabilities=["vision", "tool calling"],
    ...     is_local=False,
    ...     cost="$$$"
    ... )
    >>> print(model.id)
    gpt-4
    >>> print(model.capabilities)
    ['vision', 'tool calling']
    """

    id: str
    name: str | None = None
    description: str | None = None
    is_embedding: bool | None = None
    capabilities: list[str] | None = None
    is_local: bool | None = None
    cost: Literal["undefined", "free", "$", "$$", "$$$", "$$$$"] | None = None

    _relevant_capabilities: ClassVar[list[str]] = [
        "vision",  # Model can process images
        "usage",  # Model reports token usage in response_metadata
        "reasoning",  # Model supports reasoning_effort
        "hybrid reasoning",  # Model supports reasoning_effort as an optional variable
        "tool calling",  # Model supports tool calling a.k.a. function calling
    ]

    @model_validator(mode="before")
    @classmethod
    def flatten_meta(cls, data: Any) -> Any:
        """Flatten nested metadata structure from API responses.

        This validator processes raw API response data and extracts metadata
        from nested 'meta' or 'info.meta' structures, flattening them into
        top-level fields for easier access.

        Parameters
        ----------
        data : Any
            Raw data from API response, typically a dictionary containing
            nested metadata structures.

        Returns
        -------
        Any
            Flattened data dictionary with metadata fields extracted to
            top-level, or original data if no metadata structure found.

        Notes
        -----
        The method handles two common API response formats:
        1. Data with 'info.meta' nested structure
        2. Data with direct 'meta' field

        Extracted fields include description, capabilities, and tags which
        are then processed by field-specific validators.
        """
        if isinstance(data, dict):
            if "info" in data:
                meta = data["info"].pop("meta")
            elif "meta" in data:
                meta = data.pop("meta")
            else:
                return data
            data["description"] = meta.get("description")
            data["capabilities"] = (meta.get("capabilities") or dict()) | {
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
        """Determine if model is an embedding model from tags.

        Examines model tags to identify if the model has embedding capabilities.

        Parameters
        ----------
        v : Any
            List of tag dictionaries from model metadata.
        info : ValidationInfo
            Pydantic validation context (unused but required by validator signature).

        Returns
        -------
        bool
            True if model has 'embedding' tag, False otherwise.
        """
        tags = v or dict()
        tag_names = {t["name"].lower() for t in tags if isinstance(t, dict)}
        return "embedding" in tag_names

    @field_validator("is_local", mode="before")
    @classmethod
    def set_is_local(cls, v: Any, info: ValidationInfo):
        """Determine if model is hosted locally from tags.

        Examines model tags to identify if the model is hosted on-premises
        by Dartmouth or off-premises by a third party.

        Parameters
        ----------
        v : Any
            List of tag dictionaries from model metadata.
        info : ValidationInfo
            Pydantic validation context (unused but required by validator signature).

        Returns
        -------
        bool
            True if model has 'local' tag (case-insensitive), False otherwise.
        """
        tags = v or dict()
        tag_names = {t["name"].lower() for t in tags if isinstance(t, dict)}
        return "Local".lower() in tag_names

    @field_validator("capabilities", mode="before")
    @classmethod
    def get_capabilities(cls, v: Any, info: ValidationInfo):
        """Extract relevant capabilities from model metadata.

        Processes model capabilities and tags to create a filtered list of
        relevant capabilities as defined in ``_relevant_capabilities``.

        Parameters
        ----------
        v : Any
            Dictionary containing capabilities (as boolean flags) and tags
            (as list of dictionaries with 'name' keys).
        info : ValidationInfo
            Pydantic validation context (unused but required by validator signature).

        Returns
        -------
        list[str]
            List of capability names (lowercase) that are both enabled and
            present in ``_relevant_capabilities``.

        Notes
        -----
        Capabilities are extracted from two sources:
        1. Direct capability flags (key-value pairs where value is True)
        2. Capability tags (dictionaries with 'name' field)

        Only capabilities listed in ``_relevant_capabilities`` are included
        in the final list.
        """
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
        """Extract cost information from model tags.

        Examines model tags to determine the relative cost of using the model.

        Parameters
        ----------
        v : Any
            List of tag dictionaries from model metadata.
        info : ValidationInfo
            Pydantic validation context (unused but required by validator signature).

        Returns
        -------
        Literal["undefined", "free", "$", "$$", "$$$", "$$$$"]
            Cost indicator:
            - "free": Model is free to use
            - "$" to "$$$$": Relative cost levels (more $ = more expensive)
            - "undefined": No cost information available

        Notes
        -----
        The method searches tags in order and returns the first match:
        1. Returns "free" if a tag named "free" (case-insensitive) is found
        2. Returns the tag name if it starts with "$" (e.g., "$", "$$", etc.)
        3. Returns "undefined" if no cost information is found
        """
        tags = v
        for tag in tags:
            if tag["name"].lower() == "free":
                return "free"
            if tag["name"].startswith("$"):
                return tag["name"]
        return "undefined"


class BaseModelListing:
    """Base class for model listing interfaces.

    This abstract base class provides the common infrastructure for querying
    model listings from different sources (on-premises or cloud). It handles
    session management, authentication, and defines the interface that derived
    classes must implement.

    Parameters
    ----------
    api_key : str
        API key for authentication with the model listing service.
    url : str
        Base URL of the model listing API endpoint.

    Attributes
    ----------
    api_key : str
        Stored API key for authentication.
    SESSION : requests.Session
        HTTP session object with configured headers for API requests.
    url : str
        Base URL for API endpoints.

    Notes
    -----
    This is an abstract base class and should not be instantiated directly.
    Use :class:`DartmouthModelListing` or :class:`CloudModelListing` instead.

    Derived classes must implement:
    - :meth:`_authenticate`: Set up authentication headers for the session
    - :meth:`list`: Retrieve and return the list of available models
    """

    def __init__(self, api_key: str, url: str):
        self.api_key = api_key
        self.SESSION = requests.Session()
        self.SESSION.headers.update({"User-Agent": USER_AGENT})
        self.url = url
        self._authenticate()

    def _authenticate(self):
        """Set up authentication for API requests.

        This method must be overridden in derived classes to implement
        the specific authentication mechanism required by the API.

        Raises
        ------
        NotImplementedError
            Always raised as this is an abstract method.
        """
        return NotImplementedError

    def list(self):
        """Retrieve list of available models.

        This method must be overridden in derived classes to implement
        the specific model listing logic for the API.

        Raises
        ------
        NotImplementedError
            Always raised as this is an abstract method.
        """
        return NotImplementedError


class DartmouthModelListing(BaseModelListing):
    """Interface for listing on-premises models hosted by Dartmouth.

    This class provides access to models hosted on Dartmouth's on-premises
    infrastructure. It handles authentication using Dartmouth API keys and
    supports filtering models by various criteria.

    Parameters
    ----------
    api_key : str
        Dartmouth API key for authentication.
    url : str
        Base URL of the Dartmouth model listing API.

    Examples
    --------
    >>> from langchain_dartmouth.model_listing import DartmouthModelListing
    >>> import os
    >>> listing = DartmouthModelListing(
    ...     api_key=os.environ["DARTMOUTH_API_KEY"],
    ...     url="https://api.dartmouth.edu/models/"
    ... )
    >>> models = listing.list(type="llm")
    >>> for model in models:
    ...     print(model["id"])

    Notes
    -----
    Authentication is performed using JWT tokens obtained via the
    ``dartmouth_auth`` package. The token is automatically refreshed
    if a request fails due to authentication issues.
    """

    def _authenticate(self):
        """Set up JWT-based authentication for Dartmouth API.

        Obtains a JWT token using the Dartmouth API key and adds it
        to the session's Authorization header.
        """
        self.SESSION.headers.update(
            {"Authorization": f"Bearer {get_jwt(dartmouth_api_key=self.api_key)}"}
        )

    def list(self, **kwargs) -> List[dict]:
        """Get a list of available on-premises models.

        Retrieves models from Dartmouth's on-premises infrastructure,
        with optional filtering by server, type, or capabilities.

        Parameters
        ----------
        **kwargs : dict
            Optional filtering parameters:

            - server : str
                Filter by specific server name
            - type : str
                Filter by model type (e.g., "llm", "embedding")
            - capabilities : str or list[str]
                Filter by model capabilities

        Returns
        -------
        List[dict]
            List of model descriptions as dictionaries. Each dictionary
            contains model metadata including id, name, capabilities, etc.

        Raises
        ------
        requests.HTTPError
            If the API request fails after retry with re-authentication.

        Examples
        --------
        List all models:

        >>> models = listing.list()

        Filter by model type:

        >>> llm_models = listing.list(type="llm")

        Filter by capabilities:

        >>> vision_models = listing.list(capabilities="vision")

        Notes
        -----
        If the initial request fails, the method automatically attempts
        to re-authenticate and retry the request once.
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
    """Interface for listing cloud-based models available through Dartmouth Chat.

    This class provides access to models available through Dartmouth's cloud
    infrastructure, including both base models and customized/fine-tuned variants.
    It returns structured :class:`ModelInfo` objects with detailed metadata.

    Parameters
    ----------
    api_key : str
        API key for Dartmouth Chat authentication.
    url : str
        Base URL of the cloud model listing API.

    Examples
    --------
    >>> from langchain_dartmouth.model_listing import CloudModelListing
    >>> import os
    >>> listing = CloudModelListing(
    ...     api_key=os.environ["DARTMOUTH_CHAT_API_KEY"],
    ...     url=os.environ["LCD_CLOUD_BASE_URL"]
    ... )
    >>> models = listing.list(base_only=True)
    >>> for model in models:
    ...     print(f"{model.name} - Cost: {model.cost}")

    Notes
    -----
    Cloud models are accessed through bearer token authentication.
    The returned :class:`ModelInfo` objects provide structured access
    to model metadata including capabilities, costs, and hosting details.
    """

    def _authenticate(self):
        """Set up bearer token authentication for cloud API.

        Adds the API key as a bearer token to the session's
        Authorization header.
        """
        self.SESSION.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def list(self, base_only: bool = False) -> List[ModelInfo]:
        """Get a list of available cloud models.

        Retrieves models from Dartmouth's cloud infrastructure, with the
        option to filter for only base models or include customized variants.

        Parameters
        ----------
        base_only : bool, optional
            If True, return only base models. If False (default), return
            both base models and customized/fine-tuned variants.

        Returns
        -------
        List[ModelInfo]
            List of :class:`ModelInfo` objects containing detailed metadata
            about each available model. Only active models are included.

        Raises
        ------
        requests.HTTPError
            If the API request fails.

        Examples
        --------
        List all available models:

        >>> all_models = listing.list()

        List only base models:

        >>> base_models = listing.list(base_only=True)

        Access model details:

        >>> for model in base_models:
        ...     if "vision" in model.capabilities:
        ...         print(f"{model.name} supports vision")

        Notes
        -----
        The method automatically filters out inactive models from the results.
        Model metadata is validated and structured using the :class:`ModelInfo`
        Pydantic model, which extracts capabilities, costs, and other properties
        from the API response.
        """
        resp = self.SESSION.get(url=self.url + f"v1/models")
        resp.raise_for_status()
        cloud_models = resp.json()
        if "data" in cloud_models:
            cloud_models = cloud_models["data"]

        if base_only:

            def is_base_model(m):
                return m.get("info", {}).get("base_model_id") is None

            cloud_models = [m for m in cloud_models if is_base_model(m)]

        return [
            ModelInfo.model_validate(m)
            for m in cloud_models
            if m.get("is_active", True)
        ]


if __name__ == "__main__":
    import os

    models = CloudModelListing(
        api_key=os.environ["DARTMOUTH_CHAT_API_KEY"],
        url=os.environ["LCD_CLOUD_BASE_URL"],
    ).list(base_only=True)

    for model in models:
        print(model)
