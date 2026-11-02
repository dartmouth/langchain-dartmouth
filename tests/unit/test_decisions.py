import os
import pytest
from unittest.mock import patch
import httpx2
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAuthenticationError,
    TypeSafeNotFoundError,
    TypeSafeRateLimitError,
)
from langchain_dartmouth.decisions import DartmouthDecisions, Noul
from langchain_dartmouth.exceptions import (
    DecisionAPIError,
    DecisionRateLimitError,
    InvalidKeyError,
    ModelNotFoundError,
)

def test_dartmouth_decisions_defaults():
    decisions = DartmouthDecisions(dartmouth_chat_api_key="dummy-key")
    assert decisions.model == "cloudflare.clef"
    assert decisions.dartmouth_chat_api_key == "dummy-key"
    assert decisions.base_url.rstrip("/").endswith("chat.dartmouth.edu/api")


def test_dartmouth_decisions_user_agent():
    from langchain_dartmouth.definitions import USER_AGENT

    decisions = DartmouthDecisions(dartmouth_chat_api_key="dummy-key")
    assert decisions._request_headers["User-Agent"] == USER_AGENT


def test_dartmouth_decisions_custom_url():
    decisions = DartmouthDecisions(
        dartmouth_chat_api_key="dummy-key",
        inference_server_url="http://127.0.0.1:8080/api/",
    )
    assert decisions.base_url == "http://127.0.0.1:8080/api/"
    assert decisions._endpoint == "http://127.0.0.1:8080/api/v1/systemone"


def test_dartmouth_decisions_wraps_exceptions():
    decisions = DartmouthDecisions(dartmouth_chat_api_key="dummy-key")
    headers = httpx2.Headers({})

    with patch.object(
        decisions,
        "_classify",
        side_effect=TypeSafeAuthenticationError(401, {}, headers),
    ):
        with pytest.raises(InvalidKeyError):
            decisions.invoke({"state": "x", "questions": {}})

    with patch.object(
        decisions,
        "_classify",
        side_effect=TypeSafeNotFoundError(404, {}, headers),
    ):
        with pytest.raises(ModelNotFoundError):
            decisions.invoke({"state": "x", "questions": {}})

    with patch.object(
        decisions,
        "_classify",
        side_effect=TypeSafeRateLimitError(429, {}, headers),
    ):
        with pytest.raises(DecisionRateLimitError):
            decisions.invoke({"state": "x", "questions": {}})

    with patch.object(
        decisions,
        "_classify",
        side_effect=TypeSafeAPIConnectionError("no connection"),
    ):
        with pytest.raises(DecisionAPIError):
            decisions.invoke({"state": "x", "questions": {}})
