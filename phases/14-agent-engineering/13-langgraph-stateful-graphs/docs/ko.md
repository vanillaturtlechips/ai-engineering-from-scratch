# LangGraph: 상태 기반 그래프와 내구성 있는 실행

> LangGraph는 저수준 상태 기반 오케스트레이션을 위한 2026년 표준입니다. 에이전트는 상태 머신이며; 노드는 함수이고; 엣지는 전이(transition)이며; 상태는 불변이며 매 단계 후 체크포인트됩니다. 어떤 실패가 발생해도 정확히 중단된 지점부터 재개할 수 있습니다.

**유형:** 학습 + 구축  
**언어:** Python (표준 라이브러리)  
**선수 지식:** 14단계 · 01 (에이전트 루프), 14단계 · 12 (워크플로우 패턴)  
**소요 시간:** ~75분

## 학습 목표

- LangGraph의 핵심 모델인 **불변 상태(state machine with immutable state)**를 설명하고, **함수 노드(function nodes)**, **조건부 엣지(conditional edges)**, **포스트-스텝 체크포인트(post-step checkpoints)**를 이해한다.
- 문서에서 강조하는 4가지 기능(**지속적 실행(durable execution)**, **스트리밍(streaming)**, **인간 개입(human-in-the-loop)**, **종합적 메모리(comprehensive memory)**)을 명명한다.
- LangGraph가 지원하는 3가지 오케스트레이션 토폴로지(**감독자(supervisor)**, **피어-투-피어(군집, peer-to-peer (swarm))**, **계층적(중첩 서브그래프, hierarchical (nested subgraphs))**)를 설명한다.
- **불변 상태(stdlib state graph)**를 구현하고, **조건부 엣지(conditional edges)**와 **체크포인트/재개 주기(checkpoint/resume cycle)**를 적용한다.

## 문제

에이전트와 워크플로우는 다음과 같은 문제를 공유합니다: 40단계 실행이 38단계에서 실패했을 때, 처음부터 다시 시작하는 것이 아니라 38단계부터 재개하고 싶습니다. 2차 상태 모델은 운영자가 새 실행을 가정하는 라이브러리 주변에 재시도를 해킹하도록 강요합니다.

LangGraph의 설계 답변: 상태는 1차 타입 객체이며, 변경 사항은 명시적으로 처리되고, 모든 노드 이후에 체크포인트가 지속됩니다. 재개(resume)는 `load_state(session_id)` 호출로 이루어집니다.

## 개념

### 그래프

그래프는 다음 요소로 정의됩니다:

- **상태 유형.** 모든 노드가 읽고 변경하는 타입된 딕셔너리(또는 Pydantic 모델).
- **노드.** 순수 함수 `(state) -> state_update`. 반환 후 업데이트가 상태에 병합됨.
- **엣지.** 노드 간 조건부 또는 직접 전환.
- **진입 및 종료.** `START`와 `END` 센티넬 노드가 경계를 표시.

예시: `classify`, `refund`, `bug`, `sales`, `done` 노드를 가진 에이전트 — 그래프로 표현된 라우팅 워크플로우.

### 내구성 있는 실행

각 노드 반환 후 런타임은 상태를 직렬화하고 체커포인트 도구(SQLite, Postgres, Redis, 커스텀)에 기록합니다. N단계에서 실패 시 런타임은 `resume(session_id)`로 N+1단계부터 정확한 상태로 재개할 수 있습니다.

LangGraph 문서는 이 기능이 중요한 프로덕션 사용자(Klarna, Uber, J.P. Morgan)를 명시적으로 강조합니다. 주장은 그래프 형태가 아닌, 그래프 형태와 체커포인트 조합이 복구 비용을 줄인다는 점입니다.

### 스트리밍

모든 노드는 부분 출력을 생성할 수 있습니다. 그래프는 노드별 델타 이벤트를 호출자에게 스트리밍하여 UI가 실행 중에 업데이트되도록 합니다.

### 인간 개입

노드 간 상태를 검사하고 수정합니다. 구현: 중요 노드 전에 일시 정지, 상태를 인간에게 표시, 수정 사항 수락, 재개. 체커포인트 도구는 상태가 이미 직렬화되어 있어 이를 쉽게 만듭니다.

### 메모리

- **단기 메모리**(실행 내 — 상태 내 대화 기록)
- **장기 메모리**(실행 간 — 체커포인트 도구와 별도의 장기 저장소를 통한 지속성). LangGraph는 외부 메모리 시스템(Mem0, 커스텀)과 통합됩니다.

### 세 가지 토폴로지

1. **감독자.** 중앙 라우터 LLM이 전문가 서브에이전트에게 할당. `langgraph-supervisor`의 `create_supervisor()` (단, 2026년 LangChain 팀은 더 많은 컨텍스트 제어를 위해 직접 툴 호출을 통해 이를 구현할 것을 권장).
2. **군집 / P2P.** 에이전트들이 공유 툴 표면을 통해 직접 작업을 전달. 중앙 라우터 없음.
3. **계층적.** 하위 감독자를 관리하는 감독자, 중첩된 서브그래프로 구현.

### 이 패턴이 실패하는 경우

- **체커포인트가 너무 작음.** 대화 턴만 체커포인트하면 툴 상태와 메모리 쓰기가 복구 불가능. 전체 상태를 직렬화해야 함.
- **비결정적 노드.** 재개 시 노드 입력이 동일한 상태 업데이트를 생성한다고 가정. 난수 시드, 벽시계, 외부 API를 캡처해야 함.
- **조건부 엣지 과도 사용.** 모든 엣지가 조건부인 그래프는 추론할 수 없는 상태 머신. 가끔 분기되는 선형 체인을 선호.

## 빌드하기

`code/main.py`는 stdlib 상태 유지 그래프를 구현합니다:

- `State` — `messages`, `step`, `route`, `output`, `human_approval`을 포함하는 타입이 지정된 딕셔너리.
- `Node` — 상태를 입력으로 받아 업데이트 딕셔너리를 반환하는 호출 가능한 객체.
- `StateGraph` — 노드 + 엣지 + 조건부 엣지 + 실행 + 재개 기능.
- `SQLiteCheckpointer` (인메모리 가상) — 모든 노드 실행 후 상태를 직렬화하며, `load(session_id)`로 복원 가능.
- 데모 그래프: `classify` → 분기(`refund` / `bug` / `sales`) → 인간 승인 게이트 → `send`.

실행 방법:

```
python3 code/main.py
```

트레이스에는 첫 번째 실행이 인간 승인 게이트에서 실패한 후, 지속성(persistence)을 거쳐 재개되어 최종 출력을 생성하는 과정이 표시됩니다.

## 사용 방법

- **LangGraph** — 참조용, 프로덕션 준비 완료. `create_react_agent`, `create_supervisor` 사용 또는 직접 그래프 구축.
- **AutoGen v0.4** (레슨 14) — 고동시성 시나리오를 위한 액터 모델 대안.
- **Claude Agent SDK** (레슨 17) — 내장 세션 저장소가 있는 관리형 하니스.
- **커스텀** — 상태 형태나 체크포인터 백엔드에 대한 정확한 제어가 필요한 경우.

## Ship It

`outputs/skill-state-graph.md`는 체크포인팅과 재개 기능이 내장된 모든 대상 런타임에서 LangGraph 형태의 상태 그래프를 생성합니다.

## 연습 문제

1. 분류 신뢰도(classification confidence)가 임계값(threshold) 미만일 때 `classify`에서 `end`로 가는 조건부 엣지(conditional edge)를 추가하세요. 사람이 수동으로 `route`를 설정한 후 실행을 재개하세요.
2. SQLite 유사 가짜(fake)를 실제 SQLite 체커포인터(checkpointer)로 교체하세요. 단계별 직렬화 오버헤드(per-step serialization overhead)를 측정하세요.
3. 병렬 엣지(parallel edges)를 구현하세요: 두 노드가 동시에 실행되고, 사용자 정의 리듀서(custom reducer)로 병합합니다. 여기서 불변 상태(immutable state)가 어떤 이점을 제공하나요?
4. `langgraph-supervisor` 레퍼런스를 읽으세요. 이 예제를 `create_supervisor`로 포팅하세요. 트레이스 형태(trace shapes)를 비교하세요.
5. 스트리밍(streaming)을 추가하세요: 각 노드가 실행되는 동안 부분 상태(partial state)를 출력합니다. 델타(deltas)가 도착할 때마다 출력하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 상태 그래프 | "에이전트 상태 머신" | 타입이 지정된 상태 + 노드 + 엣지 + 리듀서 |
| 체크포인터 | "영속성 백엔드" | 모든 노드 이후 상태를 직렬화; 재개 기능 제공 |
| 리듀서 | "상태 병합기" | 현재 상태와 노드의 업데이트를 결합하는 함수 |
| 조건부 엣지 | "분기" | 상태 함수에 의해 선택되는 엣지 |
| 서브그래프 | "중첩 그래프" | 다른 그래프 내부의 노드로 사용되는 그래프 |
| 내구성 실행 | "실패 후 재개" | 정확한 상태로 마지막 성공 노드에서 재시작 |
| 슈퍼바이저 | "라우터 LLM" | 전문 서브에이전트 중앙 디스패처 |
| 스웜 | "P2P 에이전트" | 공유 도구를 통해 에이전트 간 작업 전달; 중앙 라우터 없음 |

## 추가 자료

- [LangGraph 개요](https://docs.langchain.com/oss/python/langgraph/overview) — 공식 문서
- [langgraph-supervisor 참조](https://reference.langchain.com/python/langgraph/supervisor/) — 감독자 패턴 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 액터 모델 대안
- [Claude Agent SDK 개요](https://platform.claude.com/docs/en/agent-sdk/overview) — 세션 저장소 및 서브에이전트