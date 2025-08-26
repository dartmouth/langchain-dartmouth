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
   These components provide access to the models deployed in Dartmouth's compute infrastructure using a RESTful API, and to third-party models made accessible by Dartmouth. To see which models are available, check the respective ``list()`` method of each class.


Installation
==================
You can install the latest release of the library from PyPI using pip:

.. code-block::

   pip install langchain_dartmouth

Alternatively, you can clone the `library repository <https://github.com/dartmouth/langchain-dartmouth>`_ from GitHub.

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

User Guide
======================
While this documentation contains the technical API reference, you can find a collection of tutorials (or recipes) on how to use the components in this library in the `LangChain Dartmouth Cookbook <https://dartmouth-libraries.github.io/langchain-dartmouth-cookbook/>`_.


Feedback and Comments
======================
For questions, comments, or improvements, email `Research Computing <mailto:research.computing@dartmouth.edu>`_.



How To Cite
======================
If you are using langchain_dartmouth as part of a scientific publication, we would greatly appreciate a citation of the following paper:

.. code-block:: bibtex

@inproceedings{10.1145/3708035.3736076,
author = {Stone, Simon and Crossett, Jonathan and Luker, Tivon and Leligdon, Lora and Cowen, William and Darabos, Christian},
title = {Dartmouth Chat - Deploying an Open-Source LLM Stack at Scale},
year = {2025},
isbn = {9798400713989},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3708035.3736076},
booktitle = {Practice and Experience in Advanced Research Computing 2025: The Power of Collaboration},
articleno = {43},
numpages = {5}
}


License
==================
Created by Simon Stone for Dartmouth College under `Creative Commons CC BY-NC 4.0 License <https://creativecommons.org/licenses/by/4.0/>`_

.. image:: _static/img/dartmouth-wordmark.png
   :scale: 10%
   :class: margin

.. image:: https://i.creativecommons.org/l/by/4.0/88x31.png

Except where otherwise noted, the example programs are made available under the OSI-approved MIT license.