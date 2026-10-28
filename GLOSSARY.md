# langchain_dartmouth

LangChain-compatible components that call Dartmouth-hosted models and services, wrapping or mirroring the interfaces of the upstream LangChain integrations they build on.

## Language

**Decision model**:
A model that answers one or more typed judgments about input text or structured state (binary, categorical, or ordinal), returning probabilities and confidence rather than free-form text. Dartmouth's gateway can route Decision-model requests to multiple underlying providers (e.g., `cloudflare.clef`, `typesafe.jev`).
_Avoid_: System One (TypeSafe's product name), Decisions API (OpenAI's name for a similar capability), Jev (an underlying model identifier, not the capability)
