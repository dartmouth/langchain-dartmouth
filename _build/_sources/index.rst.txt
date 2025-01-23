``langchain_dartmouth`` -- LangChain components for Dartmouth's on-premise models
=================================================================================
This package contains components to facilitate the use of models deployed in Dartmouth College's compute infrastructure or third-party models made accessible by Dartmouth. The components are fully compatible with `LangChain <https://python.langchain.com/>`_, allowing seamless integration and plug-and-play compatibility with the vast number of components in the ecosystem.

There are three main components currently implemented:

- Embedding models
   - Used to generate embeddings for text documents.
- Large Language Models:
   - Used to generate text in response to a text prompt.
- Reranking models
   - Used to rerank retrieved documents based on their relevance to a query.

.. note::
   These components provide access to the models deployed in Dartmouth's compute infrastructure using a RESTful API. To see which models are available, check the documentation of the `Dartmouth AI API <https://ai.dartmouth.edu/openapi/index.html>`_ under ``Text Generation Inference`` and ``Text Embeddings Inference``. A method to list the available models from within this library is currently in development.


Getting Started
==================
Using Dartmouth's compute infrastructure or the third-party models paid for by Dartmouth requires authentication. The components in this library handle authentication "under-the-hood", but require valid Dartmouth API keys.
For the on-premise models, you can obtain a key from `Dartmouth's Developer Portal <https://developer.dartmouth.edu/keys>`_. For the third-party models, you can find instructions on how to obtain a key `here <https://rcweb.dartmouth.edu/~d20964h/2024-12-11-dartmouth-chat-api/api_key/>`_.`

Even though you can pass your key to each component using the ``dartmouth_api_key`` or ``dartmouth_chat_api_key`` parameter, it is good practice to not include the API key in your code directly. Instead, you should set the environment variable ``DARTMOUTH_API_KEY`` or ``DARTMOUTH_CHAT_API_KEY`` to your key. This will ensure that the key is not exposed in your code.

.. note::
   We recommend using `python-dotenv <https://saurabh-kumar.com/python-dotenv/>`_ to manage your environment variables with an ``.env`` file.

.. toctree::
   api
   :maxdepth: 2
   :caption: API Reference

Feedback and Comments
======================
For questions, comments, or improvements, email `Research Computing <mailto:research.computing@dartmouth.edu>`_.


License
==================
Created by Simon Stone for Dartmouth College under `Creative Commons CC BY-NC 4.0 License <https://creativecommons.org/licenses/by/4.0/>`_

.. image:: _static/img/dartmouth-wordmark.png
   :scale: 10%
   :class: margin

.. image:: https://i.creativecommons.org/l/by/4.0/88x31.png

Except where otherwise noted, the example programs are made available under the OSI-approved MIT license.