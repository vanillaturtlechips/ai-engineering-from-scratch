# AutoGen v0.4: 액터 모델 및 에이전트 프레임워크

> AutoGen v0.4 (Microsoft Research, 2025년 1월)는 액터 모델을 중심으로 에이전트 오케스트레이션을 재설계했습니다. 비동기 메시지 교환, 이벤트 기반 에이전트, 장애 격리, 자연스러운 동시성. 이 프레임워크는 현재 유지보수 모드이며, Microsoft 에이전트 프레임워크(공개 프리뷰 2025년 10월)가 후속 버전이 됩니다.

**유형:** 학습 + 구축  
**언어:** Python (표준 라이브러리)  
**선수 지식:** 14단계 · 01 (에이전트 루프), 14단계 · 12 (워크플로우 패턴)  
**소요 시간:** ~75분

## 학습 목표

- **액터 모델** 설명: 에이전트를 액터로, 메시지를 유일한 IPC(Inter-Process Communication)로, 액터별 장애 격리.
- **AutoGen v0.4**의 세 가지 API 계층 — **Core**, **AgentChat**, **Extensions** — 과 각 계층의 목적 명명.
- **메시지 전달과 처리의 분리**가 장애 격리와 자연스러운 동시성을 제공하는 이유 설명.
- **Python으로 stdlib 액터 런타임 구현** 및 **두 에이전트 코드 리뷰 워크플로우 이식**.  

> **전문 용어**:  
> - IPC(Inter-Process Communication)  
> - stdlib(Standard Library)  

> **참고**:  
> - "Core", "AgentChat", "Extensions"는 AutoGen의 공식 API 계층명으로 번역하지 않음.  
> - "IPC"는 통신 프로토콜 용어로 원어 유지.

## 문제

대부분의 에이전트 프레임워크는 동기식입니다: 하나의 에이전트가 생산하고, 하나의 에이전트가 소비하며, 호출 스택에서 작동합니다. 실패는 스택을 충돌시킵니다. 동시성은 나중에 추가됩니다. 분산 처리를 위해서는 재작성이 필요합니다.

AutoGen v0.4의 해결책: 액터 모델. 각 에이전트는 개인 인박스를 가진 액터입니다. 메시지는 유일한 상호작용 수단입니다. 런타임은 전달과 처리를 분리합니다. 실패는 하나의 액터로 격리됩니다. 동시성은 네이티브입니다. 분산 처리는 단지 다른 전송 방식일 뿐입니다.

## 개념

### 액터

액터는 다음을 가집니다:

- **개인 상태** (외부에서 직접 접근 불가).
- **수신함** (메시지 큐).
- **핸들러**: `receive(message) -> effects` 여기서 효과는 "회신", "다른 액터에게 전송", "새 액터 생성", "상태 업데이트", "자기 자신 중지"가 될 수 있습니다.

두 액터는 메모리를 공유할 수 없습니다. 오직 메시지만 전송할 수 있습니다.

### AutoGen v0.4의 세 가지 API 계층

1. **코어.** 저수준 액터 프레임워크. `AgentRuntime`, `Agent`, `Message`, `Topic`. 비동기 메시지 교환, 이벤트 기반.
2. **에이전트 채팅.** 작업 기반 고수준 API (v0.2의 ConversableAgent 대체). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **확장.** 통합 — OpenAI, Anthropic, Azure, 도구, 메모리.

### 디커플링의 중요성

v0.2 모델에서는 `agent_a.chat(agent_b)`를 호출하면 agent_b가 응답을 반환할 때까지 agent_a가 동기적으로 차단됩니다. v0.4에서는 `send(agent_b, msg)`가 메시지를 agent_b의 수신함에 넣고 즉시 반환합니다. 런타임은 나중에 메시지를 전달합니다. 세 가지 결과:

- **장애 격리.** 에이전트 B가 충돌해도 에이전트 A가 충돌하지 않습니다 — 런타임이 B의 핸들러에서 실패를 포착하고 조치(로그, 재시도, 데드레터)를 결정합니다.
- **자연스러운 동시성.** 많은 메시지가 동시에 전송 중; 액터들은 수신함을 동시에 처리합니다.
- **분산 준비 완료.** 수신함 + 전송은 액터가 프로세스 내부 또는 다른 호스트에 있더라도 동일한 추상화입니다.

### 토폴로지

- **RoundRobinGroupChat.** 에이전트들이 고정된 순서로 번갈아가며 참여합니다.
- **SelectorGroupChat.** 선택자 에이전트가 대화 컨텍스트에 따라 다음 참여자를 선택합니다.
- **Magentic-One.** 웹 브라우징, 코드 실행, 파일 처리를 위한 참조 다중 에이전트 팀. 에이전트 채팅 기반으로 구축됨.

### 관측 가능성

OpenTelemetry 지원이 내장되어 있습니다. 모든 메시지는 스팬을 방출하며, 도구 호출은 2026 OTel GenAI 시맨틱 컨벤션(레슨 23)에 따라 `gen_ai.*` 속성을 전달합니다.

### 상태: 유지보수 모드

2026년 초: AutoGen v0.7.x는 연구 및 프로토타이핑에 안정적입니다. Microsoft는 활성 개발을 Microsoft 에이전트 프레임워크(공개 프리뷰 2025년 10월 1일; 1.0 GA는 2026년 1분기 말 목표)로 전환했습니다. AutoGen 패턴은 깨끗하게 포팅됩니다 — 액터 모델이 지속 가능한 아이디어입니다.

## 빌드하기

`code/main.py`는 표준 라이브러리 액터 런타임을 구현합니다:

- `Message` — `sender`, `recipient`, `topic`, `body`를 포함하는 타입이 지정된 페이로드.
- `Actor` — `receive(message, runtime)`을 포함하는 추상 클래스.
- `Runtime` — 공유 큐, 전달, 실패 격리를 갖춘 이벤트 루프.
- 두 액터 데모: `ReviewerAgent`는 코드를 검토하고, `ChecklistAgent`는 체크리스트를 실행하며, 합의 도출까지 메시지를 교환합니다.

실행 방법:

```
python3 code/main.py
```

트레이스에는 메시지 전달, 다른 액터를 충돌시키지 않는 한 액터의 시뮬레이션된 실패, 그리고 공유된 결론에 대한 수렴이 표시됩니다.

## 사용 방법

- **AutoGen v0.4/v0.7** (유지 관리) — 연구, 프로토타이핑, 다중 에이전트 패턴에 안정적.
- **Microsoft Agent Framework** (공개 프리뷰) — 미래 방향; 새로워진 API에서 동일한 액터 모델 아이디어 구현.
- **LangGraph 군집 토폴로지** (레슨 13) — 공유 도구 인계 방식을 통한 유사 패턴.
- **커스텀 액터 런타임** — 특정 전송 계층(NATS, RabbitMQ, gRPC)이 필요할 때 사용.

## Ship It

`outputs/skill-actor-runtime.md`는 주어진 다중 에이전트 작업을 위한 최소한의 액터 런타임과 팀 템플릿(RoundRobin 또는 Selector)을 생성합니다.

## 연습 문제

1. **데드레터 큐(DLQ) 추가**: 핸들러가 예외를 발생시킬 때, 실패한 메시지를 인간 검토를 위해 대기열에 보관하세요. 이 토이에서 DLQ는 얼마나 자주 발생하나요?  
2. **`SelectorGroupChat` 구현**: 선택자 액터가 대화 상태를 기반으로 다음 메시지를 처리할 주체를 결정하는 기능을 구현하세요.  
3. **분산 전송 추가**: 프로세스 내 큐를 JSON-over-HTTP 서버로 교체하여 액터가 별도의 프로세스에서 실행될 수 있도록 하세요.  
4. **메시지별 OTel 스팬 연결 (또는 no-op 대체)**: 레슨 23에 따라 `gen_ai.agent.name`, `gen_ai.operation.name`을(를) 발행하세요.  
5. **AutoGen v0.4 아키텍처 문서 읽기**: 이 토이를 실제 `autogen_core` API로 이식하세요. 프로덕션에서 중요한 기능 중 어떤 것을 생략했나요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Actor | "Agent" | 개인 상태 + 수신함 + 핸들러; 공유 메모리 없음 |
| Message | "Event" | 타입이 지정된 페이로드; 액터들이 상호작용하는 유일한 방법 |
| Inbox | "Mailbox" | 액터별 대기 중인 메시지 큐 |
| Runtime | "Agent host" | 메시지를 라우팅하고 장애를 격리하는 이벤트 루프 |
| Topic | "Channel" | 액터 간 명명된 발행-구독 경로 |
| Fault isolation | "Let it crash" | 한 액터의 실패가 다른 액터를 충돌시키지 않음 |
| RoundRobinGroupChat | "Fixed-rotation team" | 에이전트들이 순서대로 차례를 가짐 |
| SelectorGroupChat | "Context-routed team" | 선택기가 다음에 진행할 대상을 결정 |
| Magentic-One | "Reference team" | 웹 + 코드 + 파일을 위한 다중 에이전트 팀 |

## 추가 자료

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 재설계 포스트
- [LangGraph 개요](https://docs.langchain.com/oss/python/langgraph/overview) — 그래프 형태의 대안
- [OpenTelemetry GenAI 시맨틱 규칙](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen이 기본적으로 내보내는 스팬(span)