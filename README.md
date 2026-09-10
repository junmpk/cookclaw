# CookClaw

[中文 README](README.zh-CN.md)

CookClaw is a portfolio-oriented AI cooking assistant backend built with Python and
FastAPI. It demonstrates a production-minded Agent runtime: multi-channel request
normalization, deterministic domain handlers, hybrid recipe retrieval, durable
conversation state, guarded planning, and safe device workflows.

> Public edition: private infrastructure, credentials, login sessions, proprietary
> datasets and real hardware-provider adapters are intentionally excluded. The IoT
> boundary is represented by a local mock that never contacts real hardware.

## Why this project

The core design separates language-model reasoning from facts, authorization and
side effects:

```text
Web / IM adapters
    -> TurnApplicationService
    -> ordered TurnOrchestrator handlers
    -> bounded planner (optional, low-risk actions only)
    -> RAG / memory / mock-device ports
    -> ResponseEnvelope
    -> channel renderer
```

- LLMs interpret intent and shape natural responses.
- Recipe facts must come from the configured retrieval store.
- Device actions require explicit confirmation and deterministic state checks.
- A timed-out side effect is recorded as unknown and is never retried blindly.
- Tool evidence and state patches are separated from user-facing text.

## Highlights

- FastAPI + SSE Web API and shared QQ, WeChat and WhatsApp turn facade.
- Milvus hybrid retrieval: dense vector + BM25 + RRF + reranking.
- Search clarification, candidate selection, comparison and menu planning.
- Redis short-term task state and PostgreSQL long-term profile boundaries.
- Optional bounded planner with off, shadow and allowlisted active modes.
- Trace, replay-oriented failure datasets and regression tests.
- Local mock device adapter for safe confirmation/idempotency demonstrations.

## Repository layout

```text
app/
  api/                    FastAPI routes
  orchestrator/           turn runtime, routing and domain handlers
  conversation/           Redis/PostgreSQL state and memory adapters
  agent/                   model, retrieval and guarded-agent integration
  qqbot|weixinbot|whatsapp channel adapters
tests/                     unit and regression tests
scripts/                   local ingestion, evaluation and maintenance tools
db/                        database migrations and verification SQL
docs/                      architecture and behavior specifications
```

## Quick start

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), and a DashScope API
key. Copy the public template and keep the resulting `.env` local:

```bash
cp .env.example .env
uv sync
uv run python -m app.main
```

Then check the local health endpoint:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

Recipe retrieval requires your own authorized dataset and either Milvus Lite or a
Milvus server. No recipe workbook or production database is distributed here.

### Public web demo

The public checkout also contains a Web demo. Its left chat uses the same
conversation entry point as the QQ/WhatsApp adapters. Ordinary turns stay in the
shared runtime; an explicit request such as “ask the three experts to plan this
meal” bridges the saved task state into the LangGraph three-agent workflow. Both
paths use only the local demo database and a Mock device boundary:

```bash
uv run python -m app.demo.seed
uv run uvicorn app.demo.server:app --host 127.0.0.1 --port 8010
```

Open <http://127.0.0.1:8010>. The left rail behaves like a WhatsApp-style kitchen
chat: casual cooking questions, follow-up constraints, candidate replacement and
recipe details stay in one shared conversation. Add an explicit “three-agent
collaboration” request to see the graph nodes and reply “confirm” or “cancel” in
the same thread; before confirmation, a message such as “replace option 2” creates
a new checked graph version while preserving the other slots. The center cards display source recipe images;
if a remote image is unavailable, the card keeps a visible fallback instead of
blocking the workflow. Optional live model calls read `DASHSCOPE_API_KEY` from
the local `.env`; rehearsal mode does not call a model.

For interview demos, an optional bridge can route explicit multi-agent menu
requests from QQ, Weixin, and WhatsApp into the same LangGraph. It is disabled
by default and supports deterministic single-slot revisions such as “replace
dish 2”; see [Phase 3 IM Graph bridge](docs/im-graph-phase3.md).

## Configuration

The minimal public configuration is documented in `.env.example`. Important rules:

- never commit `.env`, channel login state, database files or device credentials;
- use `RECIPE_MILVUS_URI`, not pymilvus' reserved `MILVUS_URI` name;
- keep all real channel integrations disabled until their own credentials are set;
- the included device adapter is mock-only and cannot control real hardware.

## Verification

```bash
uv run python -m compileall -q app
uv run pytest -q
git diff --check
```

See [the architecture document](docs/architecture.md),
[Agent specifications](docs/agent-spec/README.md), and
[the public-release boundary](OPEN_SOURCE_BOUNDARY.md) for more detail.

## Project status

This repository is a technical showcase, not a hosted service or a production IoT
SDK. Real-account channel acceptance, production infrastructure, proprietary recipe
data and hardware-provider certification are outside the public repository.
