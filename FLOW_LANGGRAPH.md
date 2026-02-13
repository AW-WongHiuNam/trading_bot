# LangGraph Flow (Flow #2)

This document visualizes the current Flow #2 implemented by `run_langgraph_flow()` (and used by `run_langchain_flow()` when LangGraph is installed).

Highlights:
- Bull/Bear run a 3-round debate loop: Bull1→Bear1→Bull2→Bear2→Bull3→Bear3.
- The *tool_call loop* and *schema retry* are implemented at the **graph layer** (conditional edges) via a shared pipeline.
- Output is written as latest `flow_output_langchain.json` and archived under `outputs/<TICKER>/`.

## High-level graph

```mermaid
flowchart TD
  A[Start: run_demo.py -> run_langchain_flow()] --> G[LangGraph app.invoke(initial_state)]

  subgraph LG[run_langgraph_flow: Stage Orchestration]
    direction TB
    L[load_template]
    P[prefetch (GLOBAL_QUOTE/OVERVIEW/NEWS_SENTIMENT + trace)]

    PM[prepare_market]
    PS[prepare_social]
    PN[prepare_news]
    PF[prepare_fundamentals]

    ID[init_debate]
    PB[prepare_bull_debate]
    PR[prepare_bear_debate]
    RAB[route_after_bear]
    INC[inc_round]
    DISC[discussion]

    PRISK[prepare_risk]
    PMGR[prepare_manager]
    PTRD[prepare_trader]
    Z[finalize (write latest + archive outputs/<TICKER>/...)]

    L --> P --> PM --> PS --> PN --> PF --> ID --> PB --> PR --> RAB
    RAB -- round<3 --> INC --> PB
    RAB -- round>=3 --> DISC --> PRISK --> PMGR --> PTRD --> Z
  end

  Z --> O[Return final dict: analysts + researchers + risk + manager_decision + trader_proposal + trace]
```

## Shared work pipeline (graph-level tool_call loop + schema retry)

Every `prepare_*` node configures `work_*` fields (stage/kind/input/required/schema_hint/auto_tools/next), then hands off to this pipeline:

```mermaid
flowchart TD
  PREP[prepare_* node
  sets work_stage/work_kind/work_input
  sets required/schema_hint/auto_tools
  sets work_next] --> LLM[work_llm]

  LLM --> PARSE[work_parse
  safe JSON parse]

  PARSE -->|has tool_call && tool_iter < 2| TOOL[work_tool
  call_alpha_traced(...)
  append TOOL_RESULT
  tool_iter++]
  TOOL --> LLM

  PARSE -->|no tool_call OR tool_iter limit| VALID[work_validate
  missing = validate_schema(parsed, required)]

  VALID -->|missing empty| COMMIT[work_commit
  write parsed into state
  (analysts/bull_rounds/risk/manager/trader)]

  VALID -->|missing not empty && schema_pass==0| R1[work_repair1
  append VALIDATION_ERROR + SCHEMA_HINT
  schema_pass=1] --> LLM

  VALID -->|missing not empty && schema_pass==1| AT[work_auto_tools
  run AUTO_TOOL_CALLS
  append AUTO_TOOL_CONTEXT
  append VALIDATION_ERROR + SCHEMA_HINT
  schema_pass=2] --> LLM

  VALID -->|still missing after retries| COMMIT

  COMMIT -->|route by work_next| NEXT[Next node
  (prepare_next / route_after_bear / finalize)]
```

Notes:
- The loop limits are: `tool_iter < 2` and `schema_pass` up to 2 passes.
- `call_alpha_traced()` records tool events into `trace`.
