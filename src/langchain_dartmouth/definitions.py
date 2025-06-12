"""Configuration definitions for langchain_dartmouth.

This module contains base URLs and configuration constants used throughout
the ``langchain_dartmouth`` library. All URLs can be overridden via environment
variables.
"""

from ._version import __version__

import os

EMBEDDINGS_BASE_URL = os.environ.get(
    "LCD_EMBEDDINGS_BASE_URL", "https://api.dartmouth.edu/api/ai/tei/"
)
"""str: Base URL for the embeddings API endpoint.

Can be overridden by setting the ``LCD_EMBEDDINGS_BASE_URL`` environment variable.
Defaults to ``https://api.dartmouth.edu/api/ai/tei/``.
"""

RERANK_BASE_URL = os.environ.get(
    "LCD_RERANK_BASE_URL", "https://api.dartmouth.edu/api/ai/tei/"
)
"""str: Base URL for the reranking API endpoint.

Can be overridden by setting the ``LCD_RERANK_BASE_URL`` environment variable.
Defaults to ``https://api.dartmouth.edu/api/ai/tei/``.
"""

LLM_BASE_URL = os.environ.get(
    "LCD_LLM_BASE_URL", "https://api.dartmouth.edu/api/ai/tgi/"
)
"""str: Base URL for the Large Language Model API endpoint.

Can be overridden by setting the ``LCD_LLM_BASE_URL`` environment variable.
Defaults to ``https://api.dartmouth.edu/api/ai/tgi/``.
"""

CLOUD_BASE_URL = os.environ.get("LCD_CLOUD_BASE_URL", "https://chat.dartmouth.edu/api/")
"""str: Base URL for the Dartmouth Chat API endpoint.

Can be overridden by setting the ``LCD_CLOUD_BASE_URL`` environment variable.
Defaults to ``https://chat.dartmouth.edu/api/``.
"""

MODEL_LISTING_BASE_URL = os.environ.get(
    "LCD_MODEL_LISTINGS_BASE_URL", "https://api.dartmouth.edu/api/ai/models/"
)
"""str: Base URL for the model listings API endpoint.

Can be overridden by setting the ``LCD_MODEL_LISTINGS_BASE_URL`` environment variable.
Defaults to ``https://api.dartmouth.edu/api/ai/models/``.
"""

USER_AGENT = f"langchain_dartmouth/Python {__version__}"
