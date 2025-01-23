# Dartmouth LangChain

[![documentation](https://github.com/Dartmouth-Libraries/langchain-dartmouth/actions/workflows/docs.yml/badge.svg)](https://github.com/Dartmouth-Libraries/langchain-dartmouth/actions/workflows/docs.yml) [![tests](https://github.com/Dartmouth-Libraries/langchain-dartmouth/actions/workflows/pytest.yml/badge.svg)](https://github.com/Dartmouth-Libraries/langchain-dartmouth/actions/workflows/pytest.yml)

LangChain components for Dartmouth-hosted models.

## Getting started

1. Install the package:

```
pip install langchain_dartmouth
```

2. Obtain a Dartmouth API key from [developer.dartmouth.edu](https://developer.dartmouth.edu/)
3. Store the API key as an environment variable called `DARTMOUTH_API_KEY`:
```
export DARTMOUTH_API_KEY=<your_key_here>
```
4. Obtain a [Dartmouth Chat API key ](https://rcweb.dartmouth.edu/~d20964h/2024-12-11-dartmouth-chat-api/api_key/)
5. Store the API key as an environment variable called `DARTMOUTH_CHAT_API_KEY`
```
export DARTMOUTH_CHAT_API_KEY=<your_key_here>
```

> [!NOTE]
> You may want to[ make the environment variables permanent](https://www3.ntu.edu.sg/home/ehchua/programming/howto/Environment_Variables.html) or use a [`.env` file](https://saurabh-kumar.com/python-dotenv/)


## What is this?

This library provides an integration of Darmouth-provided generative AI resources with [the LangChain framework](https://python.langchain.com/v0.1/docs/get_started/introduction).

There are three main components currently implemented:

- Large Language Models
- Embedding models
- Reranking models

All of these components are based on corresponding LangChain base classes and can be used seamlessly wherever the corresponding LangChain objects can be used.

## Using the library

### Large Language Models

There are three kinds of Large Language Models (LLMs) provided by Dartmouth:

- On-premise:
  - Base models without instruction tuning (require no special prompt format)
  - Instruction-tuned models (also known as Chat models) requiring [specific prompt formats](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/)
- Cloud:
  - Third-party, pay-as-you-go chat models (e.g., OpenAI's GPT 4o, Google Gemini)

Using a Dartmouth-hosted base language model:

```{python}
from langchain_dartmouth.llms import DartmouthLLM

llm = DartmouthLLM(model_name="codellama-13b-hf")

response = llm.invoke("Write a Python script to swap two variables.")
print(response)
```

Using a Dartmouth-hosted chat model:

```{python}
from langchain_dartmouth.llms import ChatDartmouth


llm = ChatDartmouth(model_name="llama-3-8b-instruct")

response = llm.invoke("Hi there!")

print(response.content)
```
> [!NOTE]
> The required prompt format is enforced automatically when you are using `ChatDartmouth`.

Using a Dartmouth-provided third-party chat model:

```{python}
from langchain_dartmouth.llms import ChatDartmouthCloud


llm = ChatDartmouthCloud(model_name="openai.gpt-4o-mini-2024-07-18")

response = llm.invoke("Hi there!")


### Embeddings model

Using a Dartmouth-hosted embeddings model:

```{python}
from langchain_dartmouth.embeddings import DartmouthEmbeddingsModel


embeddings = DartmouthEmbeddingsModel()

embeddings.embed_query("Hello? Is there anybody in there?")

print(response)
```

### Reranking

Using a Dartmouth-hosted reranking model:

```{python}
from langchain_dartmouth.retrievers.document_compressors import DartmouthReranker
from langchain.docstore.document import Document


docs = [
    Document(page_content="Deep Learning is not..."),
    Document(page_content="Deep learning is..."),
    ]

query = "What is Deep Learning?"
reranker = DartmouthReranker(model_name="bge-reranker-v2-m3")
ranked_docs = reranker.compress_documents(query=query, documents=docs)

print(ranked_docs)
```


## Available models

For a list of available models, check the respective `list()` method of each class.


## License
<table >
<tbody>
  <tr>
    <td style="padding:0px;border-width:0px;vertical-align:center">
    Created by Simon Stone for Dartmouth College under <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons CC BY-NC 4.0 License</a>.<br>For questions, comments, or improvements, email <a href="mailto:researchcomputing@dartmouth.edu">Research Computing</a>.
    </td>
    <td style="padding:0 0 0 1em;border-width:0px;vertical-align:center"><img alt="Creative Commons License" src="https://i.creativecommons.org/l/by/4.0/88x31.png"/></td>
  </tr>
</tbody>
</table>

Except where otherwise noted, the example programs are made available under the OSI-approved MIT license.
