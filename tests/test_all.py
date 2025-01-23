import pytest

from langchain_dartmouth.llms import (
    DartmouthLLM,
    ChatDartmouth,
    DartmouthChatModel,
    ChatDartmouthCloud,
)
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
    assert response

    assert len(DartmouthLLM.list()) > 0


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

    assert len(ChatDartmouth.list()) > 0


def test_chat_dartmouth_cloud():
    llm = ChatDartmouthCloud(temperature=0.0, seed=42)
    response = llm.invoke("Who are you?")
    assert "OpenAI" in response.content

    llm = ChatDartmouthCloud(
        model_name="mistral.mistral-small-2409", temperature=0.0, seed=42
    )
    response = llm.invoke("Who are you?")
    assert "Mistral" in response.content

    llm = ChatDartmouthCloud(
        model_name="google_genai.gemini-1.5-pro-002", temperature=0.0, seed=42
    )
    response = llm.invoke("Who are you?")
    assert "Google" in response.content

    assert len(ChatDartmouthCloud.list()) > 0


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

    assert len(DartmouthEmbeddings.list()) > 0


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

    assert len(DartmouthReranker.list()) > 0


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


if __name__ == "__main__":
    # test_dartmouth_llm()
    # test_chat_dartmouth()
    test_chat_dartmouth_cloud()
    # test_dartmouth_chat()
    # test_dartmouth_embeddings()
    # test_dartmouth_reranker()
    # test_tei_client()   # requires locally running instance of vanilla TEI
    # # test_tei_reranker()  # requires locally running instance of vanilla TEI
