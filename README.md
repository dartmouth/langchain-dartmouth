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

## What is this?

This library provides an integration of Darmouth-hosted generative AI resources with [the LangChain framework](https://python.langchain.com/v0.1/docs/get_started/introduction).

There are three main components currently implemented:

- Large Language Models
- Embedding models
- Reranking models

All of these components are based on corresponding LangChain base classes and can be used seamlessly wherever the corresponding LangChain objects can be used.

## Using the library

### Large Language Models

There are two kinds of Large Language Models (LLMs) hosted by Dartmouth:

- Base models without instruction tuning (require no special prompt format)
- Instruction-tuned models (also known as Chat models) requiring [specific prompt formats](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/)

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
> **Note**: The required prompt format is enforced automatically when you are using `ChatDartmouth`.

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

For a list of available models, check the documentation of the RESTful [Dartmouth AI API](https://ai.dartmouth.edu/openapi/index.html).


## License
<table >
<tbody>
  <tr>
    <td style="padding:0px;border-width:0px;vertical-align:center">
    Created by Simon Stone for Dartmouth College under <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons CC BY-NC 4.0 License</a>.<br>For questions, comments, or improvements, email <a href="mailto:researchdatahelp@groups.dartmouth.edu">Research Data Services</a>.
    </td>
    <td style="padding:0 0 0 1em;border-width:0px;vertical-align:center"><img alt="Creative Commons License" src="https://i.creativecommons.org/l/by/4.0/88x31.png"/></td>
  </tr>
</tbody>
</table>

Except where otherwise noted, the example programs are made available under the OSI-approved MIT license.
