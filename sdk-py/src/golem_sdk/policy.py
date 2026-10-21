"""The exception a Golem policy block raises inside a framework adapter.

Its own module rather than a member of witness_client so an adapter can catch it
without importing the HTTP client, and so a caller can distinguish it from every
other failure by type: a Golem fault must never break a customer's chain, while a
policy refusal must. Those two go opposite ways, and telling them apart by string
matching on a message is how a refusal ends up swallowed.
"""


class GolemPolicyBlocked(Exception):
    """Raised when Golem policy refuses a tool call before it runs.

    Carries the tool name and the witness's reason separately so an adapter can
    surface the reason to the agent — an agent told only "blocked" retries, and an
    agent told why can choose a different approach.
    """

    def __init__(self, tool_name: str, reason: str = "") -> None:
        self.tool_name = tool_name
        self.reason = reason
        detail = f": {reason}" if reason else ""
        super().__init__(f"Golem policy blocked {tool_name}{detail}")


def blocked_message(reason: str) -> str:
    return f"Blocked by Golem policy: {reason}" if reason else "Blocked by Golem policy."
