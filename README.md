# golem-team-sdk

Python and TypeScript SDKs for [Golem](https://golem.team) (golem.team), AI
Detection and Response for coding agents.

Golem keeps a signed, tamper-evident record of what AI agents do and can block
the dangerous calls before they run. The Golem agent on each machine records the
coding agents it sees through their hooks and MCP traffic. These SDKs let an
agent you build yourself do the same: each tool call is hashed, signed with
COSE_Sign1 (ES256) and handed to the Golem agent, and `check_execution` can ask
it whether a call may run at all.

| | Package | Docs |
|---|---|---|
| Python | `golem-team-sdk`, imported as `golem_sdk` | [sdk-py/README.md](sdk-py/README.md) |
| TypeScript | `@golem-team/sdk` | [sdk-ts/README.md](sdk-ts/README.md) |

The Python SDK has adapters for LangChain, the OpenAI Agents SDK, CrewAI,
LlamaIndex and Semantic Kernel; each has a quickstart in
[sdk-py/docs/](sdk-py/docs/).

Both SDKs need the Golem agent running on the same machine
(`http://localhost:39393` by default). See [golem.team](https://golem.team) and
[docs.golem.team](https://docs.golem.team).

## License

Apache-2.0. See [LICENSE](LICENSE).
