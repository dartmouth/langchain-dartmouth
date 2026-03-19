Migration Guide to v0.3.x
===============

This guide helps you migrate your code from earlier versions of ``langchain_dartmouth`` to the version v0.3.0 and above. The library has undergone significant improvements to consolidate interfaces and provide better support for both on-premises and cloud-based models.

Overview of Changes
-------------------

The main changes in this version include:

1. **Unified Chat Interface**: :class:`~langchain_dartmouth.llms.ChatDartmouth` now handles both on-prem and cloud models
2. **Simplified Authentication**: Embeddings now use the same authentication system as cloud chat models and support cloud models in addition to on-prem models
3. **Improved Model Listing**: Introduction of the :class:`~langchain_dartmouth.model_listing.ModelInfo` class for type-safe model representation
4. **Deprecated Classes**: ``ChatDartmouthCloud`` is deprecated in favor of :class:`~langchain_dartmouth.llms.ChatDartmouth`

Breaking Changes
----------------

1. Chat Model Consolidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What Changed**: ``ChatDartmouthCloud`` has been deprecated and merged into :class:`~langchain_dartmouth.llms.ChatDartmouth`.

**Old Code**:

.. code-block:: python

    from langchain_dartmouth.llms import ChatDartmouth, ChatDartmouthCloud

    # For on-premise models
    on_prem_llm = ChatDartmouth(model_name="llama-3-2-11b-vision-instruct")

    # For cloud models
    cloud_llm = ChatDartmouthCloud(model_name="openai.gpt-4.1-mini-2025-04-14")

**New Code**:

.. code-block:: python

    from langchain_dartmouth.llms import ChatDartmouth

    # For on-premise models (now requires DARTMOUTH_CHAT_API_KEY)
    on_prem_llm = ChatDartmouth(model_name="meta.llama-3-2-11b-vision-instruct")

    # For cloud models (same as before)
    cloud_llm = ChatDartmouth(model_name="openai.gpt-4.1-mini-2025-04-14")

**Migration Steps**:

1. Replace all instances of ``ChatDartmouthCloud`` with :class:`~langchain_dartmouth.llms.ChatDartmouth`
2. Update your imports to remove ``ChatDartmouthCloud``
3. No other code changes are required - :class:`~langchain_dartmouth.llms.ChatDartmouth` is a drop-in replacement

2. Embeddings Authentication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What Changed**: :class:`~langchain_dartmouth.embeddings.DartmouthEmbeddings` now uses ``DARTMOUTH_CHAT_API_KEY`` instead of ``DARTMOUTH_API_KEY`` and connects to the cloud API endpoint.

**Old Code**:

.. code-block:: python

    import os
    from langchain_dartmouth.embeddings import DartmouthEmbeddings

    # Required DARTMOUTH_API_KEY environment variable
    os.environ["DARTMOUTH_API_KEY"] = "your-api-key"

    embeddings = DartmouthEmbeddings(model_name="bge-large-en-v1-5")

**New Code**:

.. code-block:: python

    import os
    from langchain_dartmouth.embeddings import DartmouthEmbeddings

    # Now requires DARTMOUTH_CHAT_API_KEY environment variable
    os.environ["DARTMOUTH_CHAT_API_KEY"] = "your-chat-api-key"

    # Model names now include provider prefix
    embeddings = DartmouthEmbeddings(model_name="baai.bge-large-en-v1-5")

**Migration Steps**:

1. Obtain a Dartmouth Chat API key from `https://chat.dartmouth.edu <https://chat.dartmouth.edu>`_
2. Replace ``DARTMOUTH_API_KEY`` with ``DARTMOUTH_CHAT_API_KEY`` in your environment variables
3. Update model names to include the provider prefix (e.g., ``"baai."`` for BGE models)
4. Update the ``dartmouth_api_key`` parameter to ``dartmouth_chat_api_key`` if passing keys directly

3. Model Listing Return Type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What Changed**: The ``list()`` method now returns a list of :class:`~langchain_dartmouth.model_listing.ModelInfo` objects instead of dictionaries.

**Old Code**:

.. code-block:: python

    from langchain_dartmouth.llms import ChatDartmouthCloud

    models = ChatDartmouthCloud.list()
    for model in models:
        print(f"Model: {model['name']}")
        print(f"Provider: {model['provider']}")

**New Code**:

.. code-block:: python

    from langchain_dartmouth.llms import ChatDartmouth

    models = ChatDartmouth.list()
    for model in models:
        print(f"Model: {model.id}")
        print(f"Name: {model.name}")
        print(f"Description: {model.description}")
        print(f"Capabilities: {model.capabilities}")
        print(f"Cost: {model.cost}")
        print(f"Is Local: {model.is_local}")

**Migration Steps**:

1. Replace dictionary access (``model['key']``) with attribute access (``model.key``)
2. Update key names: ``'name'`` → ``id``, and use new attributes like ``capabilities``, ``cost``, ``is_local``
3. Take advantage of the new structured information available in :class:`~langchain_dartmouth.model_listing.ModelInfo`

4. Default Model Names
~~~~~~~~~~~~~~~~~~~~~~

**What Changed**: Default model names have been updated to reflect the new unified interface.

**Changes**:

- :class:`~langchain_dartmouth.llms.ChatDartmouth`: Default changed from ``"llama-3-1-8b-instruct"`` to ``"openai.gpt-oss-120b"``
- :class:`~langchain_dartmouth.embeddings.DartmouthEmbeddings`: Default changed from ``"bge-large-en-v1-5"`` to ``"baai.bge-large-en-v1-5"``

**Migration Steps**:

If you were relying on default model names, explicitly specify the model you want to use:

.. code-block:: python

    # Explicitly specify the model
    llm = ChatDartmouth(model_name="meta.llama-3-1-8b-instruct")
    embeddings = DartmouthEmbeddings(model_name="baai.bge-large-en-v1-5")

New Features
------------

1. Unified Model Access
~~~~~~~~~~~~~~~~~~~~~~~

You can now access both on-prem and cloud models through a single interface:

.. code-block:: python

    from langchain_dartmouth.llms import ChatDartmouth

    # On-prem model
    local_llm = ChatDartmouth(model_name="meta.llama-3-1-8b-instruct")

    # Cloud model (OpenAI)
    openai_llm = ChatDartmouth(model_name="openai.gpt-4.1-mini-2025-04-14")

    # Cloud model (Anthropic)
    claude_llm = ChatDartmouth(model_name="anthropic.claude-4-5-sonnet-20250929")

    # Cloud model (Google)
    gemini_llm = ChatDartmouth(model_name="google_genai.gemini-2.5-flash")

2. Enhanced Model Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new :class:`~langchain_dartmouth.model_listing.ModelInfo` class provides rich information about each model:

.. code-block:: python

    from langchain_dartmouth.llms import ChatDartmouth

    models = ChatDartmouth.list()

    # Filter by capabilities
    vision_models = [m for m in models if m.capabilities and "vision" in m.capabilities]

    # Filter by cost
    free_models = [m for m in models if m.cost == "free"]

    # Filter by location
    local_models = [m for m in models if m.is_local]

3. Embedding Dimensions Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~langchain_dartmouth.embeddings.DartmouthEmbeddings` now supports specifying output dimensions for compatible models:

.. code-block:: python

    from langchain_dartmouth.embeddings import DartmouthEmbeddings

    # Use a specific dimension size (if supported by model)
    embeddings = DartmouthEmbeddings(
        model_name="openai.text-embedding-3-large",
        dimensions=256
    )

Environment Variables
---------------------

Update your environment variables as follows:

**Old**:

.. code-block:: bash

    export DARTMOUTH_API_KEY=your-api-key-here

**New**:

.. code-block:: bash

    # For on-prem models (DartmouthLLM, DartmouthReranker)
    export DARTMOUTH_API_KEY=your-api-key-here

    # For on-prem and cloud models and embeddings (ChatDartmouth, DartmouthEmbeddings)
    export DARTMOUTH_CHAT_API_KEY=your-chat-api-key-here

Complete Migration Example
---------------------------

Here's a complete example showing the migration of a typical application:

**Old Code**:

.. code-block:: python

    import os
    from langchain_dartmouth.llms import ChatDartmouth, ChatDartmouthCloud
    from langchain_dartmouth.embeddings import DartmouthEmbeddings

    # Set environment variables
    os.environ["DARTMOUTH_API_KEY"] = "your-api-key"
    os.environ["DARTMOUTH_CHAT_API_KEY"] = "your-chat-api-key"

    # Initialize models
    on_prem_chat = ChatDartmouth(model_name="llama-3-1-8b-instruct")
    cloud_chat = ChatDartmouthCloud(model_name="openai.gpt-4.1-mini-2025-04-14")
    embeddings = DartmouthEmbeddings(
        model_name="bge-large-en-v1-5",
        dartmouth_api_key=os.environ["DARTMOUTH_API_KEY"]
    )

    # Use models
    response1 = on_prem_chat.invoke("Hello!")
    response2 = cloud_chat.invoke("Hello!")
    vectors = embeddings.embed_query("Hello!")

**New Code**:

.. code-block:: python

    import os
    from langchain_dartmouth.llms import ChatDartmouth
    from langchain_dartmouth.embeddings import DartmouthEmbeddings

    # Set environment variables
    os.environ["DARTMOUTH_CHAT_API_KEY"] = "your-chat-api-key"

    # Initialize models - all use ChatDartmouth now
    on_prem_chat = ChatDartmouth(model_name="meta.llama-3-1-8b-instruct")
    cloud_chat = ChatDartmouth(model_name="openai.gpt-4.1-mini-2025-04-14")
    embeddings = DartmouthEmbeddings(
        model_name="baai.bge-large-en-v1-5",
        dartmouth_chat_api_key=os.environ["DARTMOUTH_CHAT_API_KEY"]
    )

    # Use models - same as before
    response1 = on_prem_chat.invoke("Hello!")
    response2 = cloud_chat.invoke("Hello!")
    vectors = embeddings.embed_query("Hello!")

Troubleshooting
---------------

Authentication Errors
~~~~~~~~~~~~~~~~~~~~~

If you encounter authentication errors:

1. Verify you have the correct API key for your use case:

   - ``DARTMOUTH_API_KEY`` for :class:`~langchain_dartmouth.llms.DartmouthLLM` and :class:`~langchain_dartmouth.retrievers.document_compressors.DartmouthReranker`
   - ``DARTMOUTH_CHAT_API_KEY`` for :class:`~langchain_dartmouth.llms.ChatDartmouth` and :class:`~langchain_dartmouth.embeddings.DartmouthEmbeddings`

2. Ensure your API keys are valid and not expired
3. Check that environment variables are properly set

Model Not Found Errors
~~~~~~~~~~~~~~~~~~~~~~~

If you get model not found errors:

1. Use the ``list()`` method to see available models:

   .. code-block:: python

       from langchain_dartmouth.llms import ChatDartmouth

       models = ChatDartmouth.list()
       for model in models:
           print(model.id)

2. Ensure model names include the provider prefix (e.g., ``"openai."``, ``"anthropic."``, ``"meta."``)
3. Check that the model you're trying to use is active and available

Getting Help
------------

If you encounter issues during migration:

- Check the `API Reference <api.html>`_ for detailed documentation
- Review the `LangChain Dartmouth Cookbook <https://dartmouth-libraries.github.io/langchain-dartmouth-cookbook/>`_ for examples
- Contact `Research Computing <mailto:research.computing@dartmouth.edu>`_ for support