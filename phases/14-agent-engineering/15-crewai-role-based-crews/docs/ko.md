# CrewAI: 역할 기반 크루와 플로우

> CrewAI는 2026년 역할 기반 멀티 에이전트 프레임워크입니다. 에이전트(Agents), 태스크(Tasks), 크루(Crews), 프로세스(Processes)가 4가지 기본 구성 요소입니다. 문서에서의 제작 지침: "프로덕션 준비 애플리케이션을 위해 플로우(Flow)로 시작하세요."

**유형:** 학습 + 구축  
**언어:** Python (표준 라이브러리)  
**사전 요구 사항:** 14단계 · 12 (워크플로우 패턴), 14단계 · 14 (액터 모델)  
**소요 시간:** ~60분

## 학습 목표

- CrewAI의 네 가지 기본 요소 — 에이전트(Agent), 태스크(Task), 크루(Crew), 프로세스(Process) — 의 이름을 말하고 각각의 역할을 설명할 수 있다.
- 크루(자율 역할 기반 협업)와 플로우(이벤트 기반 결정적 워크플로우)를 구분할 수 있다.
- 문서에서 프로덕션에는 플로우(Flows)를, 탐색에는 크루(Crews)를 시작하도록 권장하는 이유를 설명할 수 있다.
- stdlib 크루 러너(stdlib Crew runner)와 stdlib 플로우 러너(stdlib Flow runner)를 구현하고, 각각이 어떤 상황에서 뛰어난 성능을 보이는지 보여줄 수 있다.

## 문제

멀티 에이전트 프레임워크를 도입하는 팀들은 모두 같은 벽에 부딪힙니다: "자율적 협업"은 훌륭하게 들리지만, 고객이 버그를 보고할 때 결정적인 재생(deterministic replay)이 필요합니다. CrewAI는 이를 명시적으로 분리합니다 — 창의적인 협업을 위한 Crews, 이벤트 기반, 감사 가능, 프로덕션 환경에 적합한 워크플로우를 위한 Flows.

## 개념

### 네 가지 기본 요소

- **에이전트(Agent).** 역할(Role) + 목표(Goal) + 백스토리(Backstory) + 도구(Tools). 백스토리는 핵심 — 어조와 판단 방식을 형성합니다.
- **태스크(Task).** 설명(Description) + 예상 출력(Expected_output) + 할당 에이전트(Assigned agent). 재사용 가능한 작업 단위.
- **크루(Crew).** 에이전트와 태스크를 순차적으로 구성하는 컨테이너. 실행 프로세스(Process)를 소유합니다.
- **프로세스(Process).** 순차적(Sequential) 또는 계층적(Hierarchical, 관리자 에이전트 포함) 또는 합의형(Consensual).

### 크루 vs 플로우

- **크루(Crew).** 자율적, LLM 기반. 개방형 작업(연구, 브레인스토밍, 초안 작성)에 적합. 프레임워크가 런타임에 구조를 선택합니다.
- **플로우(Flow).** 이벤트 기반, 코드 소유 그래프. 각 단계는 트리거(함수 데코레이터, 이벤트 매칭)로 실행. 프로덕션에 적합: 관찰 가능, 테스트 가능, 결정적.

CrewAI 2026 문서에서는 "프로덕션 앱은 플로우로 시작하고, 자율성이 비용을 상쇄할 때 크루를 하위 단계로 통합하라"고 명시합니다.

### 메모리 시스템

CrewAI는 기본 제공 메모리 유형 4가지를 포함합니다: 단기(실행 내), 장기(실행 간), 엔티티(엔티티별 사실), 컨텍스트(검색 시 조립). 벡터 저장소와의 통합은 1급 지원됩니다.

### AWS 베드록 통합

CrewAI는 CloudWatch, AgentOps, Langfuse 관측 훅과 함께 AWS 베드록 통합을 문서화했습니다. AWS 문서에 따르면 QA 작업에서 LangGraph 대비 5.76배 속도 향상을 보였으나, 프레임워크별 수치는 절대적이지 않고 방향성만 참고해야 합니다.

### 의존성 구조

LangChain과 독립적입니다. Python 3.10–3.13 지원. 의존성 관리에 `uv` 사용. 2026년 초 기준 30k+ GitHub 스타.

### 이 패턴이 실패하는 경우

- **크루-프로덕션.** 플로우 래퍼 없이 프로덕션에 자유 형식 크루 사용. 출력 변동성이 높고 디버깅이 어렵습니다.
- **백스토리 과적재.** 2000단어 백스토리는 컨텍스트 예산을 초과합니다. 간결하게 유지하세요.
- **프로세스 혼동.** 계층적 프로세스는 관리자 에이전트를 추가해 라우팅합니다. 4명 이상의 전문가가 있을 때만 사용하세요.

## 빌드하기

`code/main.py`는 다음 표준 라이브러리 버전의 구현을 포함합니다:

- `Agent`, `Task`, `Crew`, `SequentialCrew`(한 번에 하나의 작업), `HierarchicalCrew`(매니저 라우팅)
- `@start()` 및 `@listen()` 데코레이터(순수 함수 대체)가 있는 `Flow`로, 명명된 이벤트에서 실행됩니다.
- 동일한 3단계 작업(연구, 개요 작성, 초안 작성)이 두 가지 방식으로 구현됩니다.

실행 방법:

```
python3 code/main.py
```

`Crew` 트레이스는 유동적이고 가변적이며, `Flow` 트레이스는 고정되고 관찰 가능합니다. 이것이 선택 사항입니다.

## 사용 방법

- **CrewAI Flow**는 프로덕션용으로 사용.
- **CrewAI Crew**는 탐색, 페어링, 초안 작성용으로 사용.
- **LangGraph** (레슨 13)는 더 명시적인 상태 머신을 원할 경우 사용.
- **AutoGen v0.4** (레슨 14)는 액터 모델 동시성을 원할 경우 사용.

## Ship It

`outputs/skill-crew-or-flow.md`은 작업에 대해 Crew vs Flow를 선택하고 최소한의 구현을 위한 구조를 제공합니다.

## 연습 문제

1. Crew 기반 데모를 Flow로 변환하세요. 변동성이 감소하는 터치포인트를 계산하세요.
2. Crew에 엔티티 메모리 추가: 고객에 대한 사실이 작업 간에 지속되도록 구현하세요.
3. 계층적 프로세스 구현: 매니저 에이전트가 이전 출력을 기반으로 다음 전문가 실행을 선택하는 구조.
4. CrewAI 문서 소개를 읽고, 토이 프로젝트를 실제 `crewai` API로 이식하세요. 테스트 가능성은 어떻게 변화하나요?
5. AgentOps 또는 Langfuse를 실행 중 하나에 연결하세요. 표준 라이브러리 버전에서 놓친 트레이스는 무엇이었나요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 에이전트(Agent) | "페르소나(Persona)" | 역할(Role) + 목표(goal) + 백스토리(backstory) + 도구(tools) |
| 태스크(Task) | "작업 단위(Unit of work)" | 설명(Description) + 예상 출력(Expected output) + 담당자(Assignee) |
| 크루(Crew) | "에이전트 팀(Agent team)" | 에이전트(Agents) + 태스크(Tasks) + 프로세스(Process)를 담는 컨테이너 |
| 프로세스(Process) | "실행 전략(Execution strategy)" | 순차적(Sequential) / 계층적(Hierarchical) / 합의형(Consensual) |
| 플로우(Flow) | "결정적 워크플로우(Deterministic workflow)" | 이벤트 기반(Event-driven), 코드 소유(Code-owned), 테스트 가능(Testable) |
| 백스토리(Backstory) | "페르소나 프롬프트(Persona prompt)" | 에이전트의 톤(Tone)과 판단(Judgment)을 형성하는 요소 |
| 엔티티 메모리(Entity memory) | "엔티티별 사실(Per-entity facts)" | 고객(Customer)/계정(Account)/이슈(Issue)별로 범위가 지정된 메모리 |

## 추가 자료

- [CrewAI 문서 소개](https://docs.crewai.com/en/introduction) — 개념 및 권장 프로덕션 경로
- [Anthropic, 효과적인 에이전트 구축](https://www.anthropic.com/research/building-effective-agents) — 멀티에이전트가 도움이 되는 경우와 그렇지 않은 경우
- [LangGraph 개요](https://docs.langchain.com/oss/python/langgraph/overview) — 상태 머신 대안