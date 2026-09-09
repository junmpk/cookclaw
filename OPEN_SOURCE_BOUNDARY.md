# Open-source boundary

This repository is a clean public snapshot. It intentionally excludes:

- Git history that previously contained retired credentials or infrastructure data;
- `.env` files, API keys, passwords, device identifiers and channel login sessions;
- private server addresses, tunnels, deployment runbooks and operational reports;
- proprietary recipe workbooks, generated databases and production traces;
- real hardware-provider endpoints, authentication flows and command protocols;
- local IDE, agent-workflow, cache, build and release artifacts.

The included device scripts are local mocks. They preserve the application boundary
for demonstrations and tests but cannot authenticate to or control real hardware.

Before contributing data or a new integration, verify that you have the right to
publish it and add only redacted examples or explicit test fixtures.

