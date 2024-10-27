import pytest

from langchain_dartmouth.llms import DartmouthLLM, ChatDartmouth, DartmouthChatModel
from langchain_dartmouth.embeddings import DartmouthEmbeddings
from langchain_dartmouth.cross_encoders import TextEmbeddingInferenceClient
from langchain_dartmouth.retrievers.document_compressors import (
    TeiCrossEncoderReranker,
    DartmouthReranker,
)

from langchain.docstore.document import Document
from langchain.schema import SystemMessage, HumanMessage

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def test_dartmouth_llm():
    llm = DartmouthLLM()
    response = llm.invoke("Write a Python script to swap the values of two variables")
    print(response)


def test_chat_dartmouth():
    llm = ChatDartmouth(model_name="llama-3-8b-instruct")
    response = llm.invoke("Please respond with the single word OK")
    assert response.content.strip() == "OK"

    llm = ChatDartmouth(model_name="llama-3-1-8b-instruct")
    response = llm.invoke(
        [
            SystemMessage(content="You are a cat."),
            HumanMessage(content="What is your name?"),
        ]
    )
    assert response.content


def test_dartmouth_chat():
    llm = DartmouthChatModel(model_name="llama-3-8b-instruct", temperature=0.01)
    response = llm.invoke(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease respond with the single word OK<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    assert response.strip() == "OK"

    llm = DartmouthChatModel(model_name="codellama-13b-instruct-hf")
    response = llm.invoke("<s>[INST]Please respond with the single word OK[/INST]")
    assert response.strip() == "OK"

    llm = DartmouthChatModel(
        inference_server_url="https://ai-api.dartmouth.edu/tgi/codellama-13b-instruct-hf/",
    )
    print(llm.invoke("<s>[INST]Hello[/INST]"))


def test_dartmouth_embeddings():
    embeddings = DartmouthEmbeddings()
    result = embeddings.embed_query("Is there anybody out there?")
    assert result

    embeddings = DartmouthEmbeddings(
        jwt_url="https://api-dev.dartmouth.edu/api/jwt",
        embeddings_server_url="https://ai-api-dev.dartmouth.edu/tei/",
    )
    result = embeddings.embed_query("Is there anybody out there?")
    assert result


def test_dartmouth_reranker():
    docs = [
        Document(page_content="Deep Learning is not..."),
        Document(page_content="Deep learning is..."),
    ]
    query = "What is Deep Learning?"
    reranker = DartmouthReranker()
    ranked_docs = reranker.compress_documents(query=query, documents=docs)
    assert ranked_docs

    reranker = DartmouthReranker(top_n=1)
    ranked_docs = reranker.compress_documents(query=query, documents=docs)
    assert len(ranked_docs) == 1

    reranker = DartmouthReranker(
        top_n=1,
        jwt_url="https://api-dev.dartmouth.edu/api/jwt",
        embeddings_server_url="https://ai-api-dev.dartmouth.edu/tei/",
    )
    ranked_docs = reranker.compress_documents(query=query, documents=docs)
    assert len(ranked_docs) == 1


@pytest.mark.skip(reason="Needs a locally running instance of TEI")
def test_tei_reranker():
    docs = [
        Document(page_content="Deep Learning is not..."),
        Document(page_content="Deep learning is..."),
    ]
    query = "What is Deep Learning?"
    cross_encoder = TeiCrossEncoderReranker()
    ranked_docs = cross_encoder.compress_documents(query=query, documents=docs)

    assert ranked_docs


@pytest.mark.skip(reason="Needs a locally running instance of TEI")
def test_tei_client():
    query = "What is Deep Learning?"
    texts = [
        "Deep Learning is not...",
        "Deep learning is...",
    ]
    tei_client = TextEmbeddingInferenceClient()
    scores = tei_client.rerank(query=query, texts=texts)

    assert scores


def test_tool_use():
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiplies a and b."""
        return a * b

    tools = [add, multiply]

    llm = ChatDartmouth()
    llm_with_tools = llm.bind_tools(tools)

    from langchain_core.messages import HumanMessage

    query = "What is 3 * 12? Also, what is 11 + 49?"

    messages = [HumanMessage(query)]

    ai_msg = llm_with_tools.invoke(messages)
    ai_msg.content = ""
    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)

    for message in messages:
        message.pretty_print()

    llm_with_tools.invoke(messages).pretty_print()


if __name__ == "__main__":
    # test_dartmouth_llm()
    # test_chat_dartmouth()
    # test_dartmouth_chat()
    # test_dartmouth_embeddings()
    # test_dartmouth_reranker()
    test_tool_use()
    # test_tei_client()   # requires locally running instance of vanilla TEI
    # # test_tei_reranker()  # requires locally running instance of vanilla TEI
