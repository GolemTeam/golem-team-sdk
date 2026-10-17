"""Framework-specific instrumentation adapters for the Golem SDK.

Each submodule wires a popular agent framework's callback / observer
surface into ``GolemSDK.intercept_execution`` so customers can adopt
Golem without rewriting their tool-call sites by hand.

Submodules are import-time optional — they are only meaningful when
the corresponding framework is installed (e.g. ``pip install
golem-team-sdk[langchain]`` or ``pip install golem-team-sdk[openai-agents]``).
The base ``golem_sdk`` package never imports any of them automatically.
"""
