# Pin langchain-typesafe to an exact alpha version

`DartmouthDecisions` (issue #55) builds on `langchain-typesafe`, which is published as a pre-release (`0.0.1a3`) with no stability guarantees between alpha builds. We pin the exact version (`==0.0.1a3`) in `pyproject.toml` rather than a range, because an unpinned alpha could silently change field names, response shapes, or exception types out from under `DartmouthDecisions` on a routine `uv sync`. Bumping the pin is a deliberate, reviewed action; it is not automatic.
