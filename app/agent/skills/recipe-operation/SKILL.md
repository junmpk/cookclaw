---
name: recipe-operation
description: Local mock device adapter for demonstrating CookClaw's deterministic device workflow.
---

# Mock recipe operation

This public adapter never contacts real hardware or a vendor cloud. It preserves the
CLI boundary used by the orchestrator so the confirmation, idempotency, timeout and
state-verification paths can be demonstrated safely on a local machine.

