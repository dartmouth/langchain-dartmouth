from typing import Any
from langchain_core.messages import AIMessage


def add_response_cost_to_usage_metadata(response: AIMessage) -> AIMessage:
    """Add the response cost from the header to the usage metadata"""
    response_cost = response.response_metadata["headers"].get("x-litellm-response-cost")
    if response_cost:
        response.usage_metadata["response_cost"] = response_cost
    return response


def filter_dartmouth_chat_models(
    models: list[dict[str, Any]],
    include_tag: str | list[str] | None = None,
    exclude_tag: str | list[str] | None = None,
    include_capability: str | list[str] | None = None,
    exclude_capability: str | list[str] | None = None,
    active_only: bool = True,
) -> list[dict[str, Any]]:

    if active_only:
        models = [m for m in models if m.get("is_active", True)]

    def _has_tag(model, tag):
        model_tags = model.get("meta", {}).get("tags")
        if not model_tags:
            return False

        for model_tag in model_tags:
            if tag.lower().strip() == model_tag["name"].lower().strip():
                return True
        return False

    def _has_capability(model, capability):
        model_capabilities = model.get("meta", {}).get("capabilities")
        if capability.lower() in model_capabilities.keys():
            return model[model_capabilities[capability]]
        return False

    if include_tag is not None:
        if isinstance(include_tag, str):
            include_tag = [include_tag]
        for tag in include_tag:
            models = [m for m in models if _has_tag(m, tag=tag)]

    if exclude_tag is not None:
        if isinstance(exclude_tag, str):
            exclude_tag = [exclude_tag]
        for tag in exclude_tag:
            models = [m for m in models if not _has_tag(m, tag=tag)]

    if include_capability is not None:
        if isinstance(include_capability, str):
            include_capability = [include_capability]
        for capability in include_capability:
            models = [m for m in models if _has_capability(m, capability=capability)]

    if exclude_capability is not None:
        if isinstance(exclude_capability, str):
            exclude_capability = [exclude_capability]
        for tag in exclude_capability:
            models = [
                m for m in models if not _has_capability(m, capability=capability)
            ]

    return models
