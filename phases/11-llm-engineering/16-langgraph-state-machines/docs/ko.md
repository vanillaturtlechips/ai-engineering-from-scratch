# LangGraph — 에이전트를 위한 상태 머신

> 직접 작성한 ReAct 루프는 `while True`입니다. LangGraph로 작성한 ReAct 루프는 체크포인트, 중단, 분기, 시간 이동이 가능한 그래프입니다. 에이전트는 변하지 않았습니다. 그 주변을 감싼 하네스가 달라졌을 뿐입니다.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** 11단계 · 09 (함수 호출), 11단계 · 14 (모델 컨텍스트 프로토콜)
**소요 시간:** ~75분

## 문제

함수 호출 에이전트를 출시합니다. 3턴 동안 작동하다가 문제가 발생합니다: 모델이 500 오류를 반환하는 도구를 시도하거나, 사용자가 작업 중간에 마음을 바꾸거나, 에이전트가 인간의 승인 없이 주문을 환불하기로 결정하는 경우입니다. `while True:` 루프에는 훅이 없습니다. 일시 중지할 수 없고, 되돌릴 수 없으며, "모델이 다른 도구를 선택했다면 어떻게 될까?"라는 분기를 만들 수도 없습니다. 데모를 넘어서 출시하는 순간, 에이전트는 작동했거나 작동하지 않은 블랙박스가 되어 버립니다.

다음 단계는 한눈에 보면 명확합니다. 에이전트는 이미 상태 머신입니다 — 시스템 프롬프트 + 메시지 기록 + 보류 중인 도구 호출 + 다음 액션. 상태 머신을 명시적으로 만드세요: "모델의 사고", "도구 실행", "인간 승인"을 위한 노드와 이들 간의 조건부 전이를 위한 엣지를 추가합니다. 그래프가 명시적으로 표현되면 하네스는 네 가지 기능을 무료로 얻습니다: 체크포인팅(단계 간 상태 저장), 인터럽트(인간 개입을 위한 일시 중지), 스트리밍(토큰 및 중간 이벤트 스트리밍), 시간 여행(이전 상태로 되돌린 후 다른 분기 시도).

LangGraph는 이 추상화를 제공하는 라이브러리입니다. LangChain 방식의 에이전트 프레임워크("여기 AgentExecutor가 있습니다. 행운을 빕니다")가 아닙니다. 1급 상태, 1급 지속성, 1급 인터럽트를 갖춘 그래프 런타임입니다. 에이전트 루프는 직접 그리는 것이지, 직접 손으로 작성하는 것이 아닙니다.

## 개념

![LangGraph StateGraph: 노드, 엣지, 체크포인터](../assets/langgraph-stategraph.svg)

`StateGraph`는 세 가지 요소로 구성됩니다.

1. **상태(State).** 그래프를 통해 흐르는 타입이 지정된 딕셔너리(TypedDict 또는 Pydantic 모델)입니다. 모든 노드는 전체 상태를 입력받고 부분 업데이트를 반환하며, LangGraph는 필드별 *리듀서*를 사용해 병합합니다. 리스트 누적에는 `operator.add`를 사용하고, 기본값은 덮어쓰기입니다.
2. **노드(Nodes).** 파이썬 함수 `state -> partial_state`입니다. 각각은 이산적인 단계(예: "모델 호출", "툴 실행", "요약")를 나타냅니다.
3. **엣지(Edges).** 노드 간 전환입니다. 정적 엣지는 한 방향으로 이동합니다. 조건부 엣지는 라우터 함수 `state -> next_node_name`을 사용해 모델 출력에 따라 분기합니다.

그래프를 컴파일합니다. 컴파일은 토폴로지를 바인딩하고 체크포인터를 연결(선택 사항이지만 프로덕션에 필수)하며 실행 가능한 객체를 반환합니다. 초기 상태와 `thread_id`로 호출합니다. 실행의 모든 단계는 `(thread_id, checkpoint_id)`를 키로 체크포인트를 지속합니다.

### 네 가지 핵심 기능

**체크포인팅(Checkpointing).** 모든 노드 전환은 상태를 저장소(테스트 시 메모리, 프로덕션 시 Postgres/Redis/SQLite)에 기록합니다. 동일한 `thread_id`로 그래프를 다시 호출해 재개할 수 있습니다. 그래프는 일시 중지된 지점부터 계속 실행됩니다.

**인터럽트(Interrupts).** `interrupt_before=["human_review"]`로 노드를 표시하면 해당 노드 실행 전에 중단됩니다. 상태는 지속됩니다. API는 사용자에게 "승인 대기 중"으로 응답합니다. 이후 `Command(resume=...)`와 함께 동일한 `thread_id`로 요청하면 실행이 재개됩니다.

**스트리밍(Streaming).** `graph.stream(state, mode="updates")`는 상태 델타를 실시간으로 제공합니다. `mode="messages"`는 모델 노드 내 LLM 토큰을 스트리밍합니다. `mode="values"`는 전체 스냅샷을 제공합니다. UI에 표시할 내용을 선택할 수 있습니다.

**타임 트래블(Time-travel).** `graph.get_state_history(thread_id)`는 전체 체크포인트 로그를 반환합니다. 이전 `checkpoint_id`를 `graph.invoke`에 전달하면 해당 시점부터 분기됩니다. 디버깅("모델이 툴 B를 선택했다면?")과 프로덕션 트레이스 재생 테스트에 유용합니다.

### 리듀서가 핵심입니다

모든 상태 필드에는 리듀서가 있습니다. 대부분 기본값이 적합합니다. 새 값이 이전 값을 덮어씁니다. 그러나 메시지 리스트는 `operator.add`가 필요해 새 메시지가 추가되도록 합니다. 병렬 엣지는 리듀서를 통해 업데이트를 병합합니다. 두 노드가 모두 `messages`를 업데이트하는데 `Annotated[list, add_messages]`를 잊으면 두 번째 업데이트가 조용히 우선권을 가져 한 턴의 절반이 손실됩니다. 리듀서는 라이브러리에서 유일한 미묘한 부분입니다. 올바르게 설정하면 나머지는 조합 가능합니다.

### 네 노드로 구성된 ReAct 그래프

프로덕션 ReAct 에이전트는 네 노드와 두 엣지로 구현됩니다:

1. `agent` — 현재 메시지 기록으로 LLM을 호출합니다. 어시스턴트 메시지(툴 호출 포함 가능)를 반환합니다.
2. `tools` — 마지막 어시스턴트 메시지의 툴 호출을 실행하고, 툴 결과를 툴 메시지로 추가합니다.
3. `agent`에서 출발하는 조건부 엣지 — 마지막 메시지에 툴 호출이 있으면 `tools`로, 없으면 `END`로 라우팅합니다.
4. `tools`에서 `agent`로 향하는 정적 엣지.

이것이 전부입니다. 체크포인팅, 인터럽트, 스트리밍을 갖춘 완전한 ReAct 루프(생각 → 행동 → 관찰 → 생각 → …)를 약 40줄의 코드로 구현할 수 있습니다.

### StateGraph vs Send (팬아웃)

`Send(node_name, state)`는 노드가 병렬 서브그래프를 디스패치할 수 있게 합니다. 예시: 에이전트가 세 개의 리트리버를 동시에 쿼리합니다. 각 `Send`는 대상 노드의 병렬 실행을 생성하며, 출력은 상태 리듀서를 통해 병합됩니다. 이는 LangGraph가 스레딩 프리미티브 없이 오케스트레이터-워커 패턴을 표현하는 방식입니다.

### 서브그래프

컴파일된 그래프는 다른 그래프의 노드가 될 수 있습니다. 외부 그래프는 단일 노드를 인식하며, 내부 그래프는 자체 상태와 체크포인트를 가집니다. 이는 팀이 감독자-워커 에이전트를 구축하는 방식입니다. 감독자 그래프는 사용자 의도를 도메인별 워커 서브그래프로 라우팅합니다.

## 빌드하기

### 1단계: 상태와 노드

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages`는 메시지 목록이 덮어쓰기 대신 누적되도록 하는 리듀서입니다. 이를 잊는 것이 가장 흔한 LangGraph 버그입니다.

### 2단계: 스레드로 실행

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

모든 업데이트는 `{node_name: state_delta}` 형식의 딕셔너리입니다. 프론트엔드는 이를 UI에 스트리밍하여 사용자가 "에이전트 생각 중… 검색 웹 호출… 결과 획득… 답변 중"과 같은 상태를 볼 수 있도록 할 수 있습니다.

### 3단계: 인간 개입 인터럽트 추가

노드에 표시를 추가하여 실행 전에 일시 중지합니다.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # 모든 도구 호출 전에 일시 중지
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"]가 설정됨. 제안된 도구 호출 검사.
# 승인된 경우:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# 거부된 경우: 거부 메시지 작성 후 재개
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

상태, 체크포인트, 스레드는 모두 인터럽트 간에 지속됩니다. 실행 중을 제외하고는 메모리에 아무것도 남지 않습니다.

### 4단계: 디버깅을 위한 시간 여행

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# 이전 체크포인트에서 분기
target = history[3].config  # 3단계 전으로 이동
for event in app.stream(None, target, stream_mode="values"):
    pass  # 해당 지점부터 재생
```

입력을 `None`으로 전달하면 주어진 체크포인트부터 재생됩니다. 값을 전달하면 해당 체크포인트 상태에 업데이트로 추가된 후 재개됩니다. 이는 전체 대화를 다시 실행하지 않고 잘못된 에이전트 실행을 재현하는 방법입니다.

### 5단계: 프로덕션을 위한 체크포인터 교체

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis, Postgres가 기본 제공됩니다. `MemorySaver`는 테스트용입니다. 재시작 시 지속되어야 하는 모든 것은 실제 저장소가 필요합니다.

## 기술

> `while True` 루프가 아닌 그래프로 에이전트를 구축합니다.

LangGraph를 사용하기 전에 60초 설계 검토를 수행하세요:

1. **노드 이름 지정.** 모든 이산적 결정 또는 부작용이 있는 작업은 노드입니다. "에이전트 사고", "툴 실행", "검토자 승인", "응답 스트리밍" 등. 목록을 작성할 수 없다면 아직 에이전트 형태가 아닙니다.
2. **상태 선언.** 모든 리스트 필드에 리듀서가 있는 최소한의 TypedDict. 모든 것을 `messages`에 넣지 마세요. 작업별 필드(작업 중인 `plan`, `budget` 카운터, `retrieved_docs` 리스트 등)를 최상위 레벨로 승격시키세요.
3. **에지 그리기.** 다음 단계가 모델 출력에 의존하지 않는 한 정적입니다. 모든 조건부 에지에는 명명된 브랜치가 있는 라우터 함수가 필요합니다.
4. **체크포인터를 미리 선택.** 테스트에는 `MemorySaver`, 그 외 모든 경우에는 Postgres/Redis/SQLite. 체크포인터가 없으면 출시하지 마세요 — 체크포인터가 없으면 재개, 중단, 시간 이동이 불가능합니다.
5. **툴 실행 전에 중단 결정.** 부작용은 에지로 이동하여 노드 진입 전에 취소할 수 있도록 하고, 검증은 모델 출력 에지에 배치하여 저비용으로 잘못된 호출을 거부할 수 있게 합니다.
6. **기본적으로 스트리밍.** UI에는 `mode="updates"`, 모델 노드 내 토큰 수준 스트리밍에는 `mode="messages"`, 평가 중 전체 스냅샷에는 `mode="values"`를 사용합니다.

체크포인터가 없는 LangGraph 에이전트는 출시하지 마세요. 부작용 *후*에 중단하는 에이전트는 출시하지 마세요. `add_messages` 리듀서가 없는 `messages` 필드는 출시하지 마세요.

## 연습 문제

1. **쉬움.** 위의 4노드 ReAct 그래프를 계산기 도구(calculator tool)와 웹 검색 도구(web-search tool)로 구현하세요. `list(app.get_state_history(config))`가 2턴 대화에서 최소 4개의 체크포인트를 반환하는지 검증하세요.
2. **중간.** `agent` 이전에 실행되는 `planner` 노드를 추가하고, 구조화된 `plan: list[str]`을 상태에 기록하세요. `agent`가 계획 단계를 완료 처리하도록 구현하세요. 체크포인트 재개 시 `plan`이 손실되면(잘못된 리듀서) 테스트를 실패 처리하세요.
3. **어려움.** `Send`를 사용하여 세 개의 서브그래프(`researcher`, `writer`, `reviewer`) 간 라우팅을 수행하는 감독자(supervisor) 그래프를 구축하세요. 각 서브그래프는 자체 상태와 체크포인터를 가져야 합니다. 외부 그래프에 `interrupt_before=["writer"]`를 추가하여 사람이 연구 개요(research brief)를 승인할 수 있도록 하세요. 이전 체크포인트에서 시간 이동(time-travel) 시 분기된(branched) 하위 그래프만 다시 실행되는지 확인하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| StateGraph | "The LangGraph graph" | 컴파일 전에 노드와 엣지를 추가하는 빌더 객체. |
| Reducer | "How the field merges" | 노드가 해당 필드에 대한 업데이트를 반환할 때 적용되는 `(old, new) -> merged` 함수; 기본값은 덮어쓰기, `add_messages`는 추가. |
| Thread | "A conversation ID" | 하나의 세션에 대한 모든 체크포인트를 스코프하는 `thread_id` 문자열. |
| Checkpoint | "A paused state" | 노드 전환 후 전체 그래프 상태의 지속된 스냅샷, `(thread_id, checkpoint_id)`로 키 관리. |
| Interrupt | "Pause for a human" | `interrupt_before` / `interrupt_after`는 노드 경계에서 실행을 중지; `Command(resume=...)`로 재개. |
| Time-travel | "Fork from a prior step" | `graph.invoke(None, config_with_old_checkpoint_id)`는 해당 체크포인트부터 재실행. |
| Send | "Parallel subgraph dispatch" | 노드가 N개의 병렬 실행을 대상 노드에 생성하기 위해 반환할 수 있는 생성자. |
| Subgraph | "A compiled graph as a node" | 다른 그래프의 노드로 사용되는 컴파일된 StateGraph; 자체 상태 스코프 유지.

## 추가 학습 자료

- [LangGraph 문서](https://langchain-ai.github.io/langgraph/) — StateGraph, 리듀서(reducer), 체크포인팅(checkpointer), 인터럽트(interrupt)에 대한 공식 참조 자료.
- [LangGraph 개념: 상태, 리듀서, 체크포인팅](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 이 레슨에서 사용하는 정신 모델, 공식 출처에서 직접 제공.
- [LangGraph 지속성 및 체크포인트](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis 저장소, 체크포인트 네임스페이스, 스레드 ID에 대한 상세 설명.
- [LangGraph 인간-루프(Human-in-the-loop)](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`, `interrupt_after`, `Command(resume=...)`, 상태 편집 패턴.
- [Yao et al., "ReAct: 언어 모델에서 추론과 행동 통합" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 모든 LangGraph 에이전트가 구현하는 패턴; 추론 추적 근거를 위해 필독.
- [Anthropic — 효과적인 에이전트 구축 (2024년 12월)](https://www.anthropic.com/research/building-effective-agents) — 체인(chain), 라우터(router), 오케스트레이터-워커(orchestrator-workers), 평가자-최적화기(evaluator-optimizer) 중 어떤 그래프 형태를 언제 선호해야 하는지.
- Phase 11 · 09 (함수 호출) — 모든 LangGraph 에이전트 노드가 재사용하는 도구 호출 기본 요소.
- Phase 11 · 14 (모델 컨텍스트 프로토콜) — MCP 어댑터를 통해 LangGraph `ToolNode`에 연결되는 외부 도구 검색.
- Phase 11 · 17 (에이전트 프레임워크 트레이드오프) — CrewAI, AutoGen, Agno 대신 LangGraph를 선택해야 하는 경우.