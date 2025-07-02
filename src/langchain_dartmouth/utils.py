from langchain_core.messages import AIMessage


def add_response_cost_to_usage_metadata(response: AIMessage) -> AIMessage:
    """Add the response cost from the header to the usage metadata"""
    response_cost = response.response_metadata["headers"]["x-litellm-response-cost"]
    response.usage_metadata["response_cost"] = response_cost
    return response
