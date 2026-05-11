# A2A — 에이전트 간 프로토콜

> Google은 2025년 4월 A2A를 발표했으며, 2026년 4월까지 사양은 https://a2a-protocol.org/latest/specification/에 공개되었고 150개 이상의 조직이 이를 지원하고 있습니다. A2A는 MCP(레슨 13)의 수평적 보완 프로토콜입니다: MCP가 수직적(에이전트 ↔ 도구) 연결이라면, A2A는 P2P(에이전트 ↔ 에이전트) 연결을 위한 것입니다. 에이전트 카드(발견), 아티팩트(텍스트, 구조화 데이터, 비디오)를 포함한 작업, 불투명한 작업 라이프사이클, 인증 등을 정의합니다. 프로덕션 시스템에서는 MCP와 A2A를 함께 사용하는 사례가 증가하고 있습니다. Google Cloud는 2025-2026년 동안 Vertex AI 에이전트 빌더에 A2A 지원을 통합했습니다.

**유형:** 학습 + 실습  
**언어:** Python (표준 라이브러리, `http.server`, `json`)  
**선수 조건:** 16단계 · 04 (기본 모델)  
**소요 시간:** ~75분

## 문제

에이전트가 다른 시스템의 에이전트를 호출해야 하는 경우 어떻게 해야 할까요? HTTP 엔드포인트를 노출하고, 맞춤형 JSON 스키마를 정의한 후 상대방이 이를 지원하기를 바랄 수 있습니다. 하지만 이 방식은 모든 에이전트 쌍마다 별도의 통합 작업이 필요합니다.

A2A(Agent-to-Agent)는 이러한 호출을 위한 범용 와이어 프로토콜입니다. 표준 발견(standard discovery), 표준 작업 모델(standard task model), 표준 전송(standard transport), 표준 아티팩트(standard artifacts)를 제공합니다. HTTP+REST가 웹을 위한 표준인 것처럼, A2A는 에이전트를 1급 시민으로 취급하는 표준 프로토콜입니다.

## 개념

### 네 가지 요소

**에이전트 카드(Agent Card).** `/.well-known/agent.json`에 있는 JSON 문서로 에이전트 설명: 이름, 기술, 엔드포인트, 지원 모달리티, 인증 요구사항. 카드의 내용을 읽어 디스커버리를 수행합니다.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**태스크(Task).** 작업 단위. 비동기적, 상태(state)를 가진 객체로 수명 주기: `제출됨(submitted) → 작업 중(working) → 완료됨(completed) / 실패(failed) / 취소됨(canceled)`. 클라이언트가 태스크를 전송하고 업데이트를 폴링하거나 구독합니다.

**아티팩트(Artifact).** 태스크가 생성하는 결과 유형. 텍스트, 구조화된 JSON, 이미지, 비디오, 오디오. 아티팩트는 타입이 지정되어 있어 다양한 모달리티가 1급(first-class) 시민입니다.

**불투명 수명 주기(Opaque lifecycle).** A2A는 원격 에이전트가 태스크를 해결하는 *방법*을 규정하지 않습니다. 클라이언트는 상태 전이와 아티팩트만 확인하며, 구현은 어떤 프레임워크든 자유롭게 사용할 수 있습니다.

### MCP/A2A 분리

- **MCP** (레슨 13): 에이전트 ↔ 도구. 에이전트는 JSON-RPC를 통해 도구 서버에 읽기/쓰기를 수행합니다. 기본적으로 상태 없음(stateless).
- **A2A**: 에이전트 ↔ 에이전트. 피어 프로토콜; 양측 모두 자체 추론 능력을 가진 에이전트입니다.

실제 다중 에이전트 시스템은 둘 다 사용합니다. A2A 피어는 자체 측에서 MCP 도구를 호출합니다. 이 분리는 두 관심사를 깨끗하게 유지합니다.

### 디스커버리 흐름

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──에이전트 카드 JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

또는 스트리밍: `/tasks/{id}/events`에 대한 SSE 구독을 통해 푸시 업데이트 수신.

### 인증(Auth)

A2A는 세 가지 일반적인 패턴을 지원합니다:

- **베어러 토큰(Bearer token)** — OAuth2 또는 불투명 토큰.
- **mTLS** — 상호 TLS; 조직이 서로에게 신원을 증명.
- **서명된 요청(Signed requests)** — 페이로드에 대한 HMAC.

인증은 에이전트 카드에 선언되며, 클라이언트는 이를 발견하고 준수합니다.

### 2026년 4월 기준 150+ 조직

엔터프라이즈 채택이 A2A 확장을 주도했습니다. 핵심 내용: A2A는 엔터프라이즈 에이전트 시스템이 신뢰 경계를 넘는 방식이 되었습니다. Google Cloud는 Vertex AI 에이전트 빌더에 A2A 지원을 출시했고, Microsoft 에이전트 프레임워크도 지원하며, 대부분의 주요 프레임워크(LangGraph, CrewAI, AutoGen)는 A2A 어댑터를 제공합니다.

### A2A가 강점을 보이는 분야

- **조직 간 호출.** 회사 A의 에이전트가 회사 B의 에이전트를 호출. A2A가 없으면 모든 쌍이 맞춤형 계약입니다.
- **이기종 프레임워크.** LangGraph 에이전트가 CrewAI 에이전트를 호출하고, 이는 커스텀 Python 에이전트를 호출. A2A가 표준화합니다.
- **타입이 지정된 아티팩트.** 비디오 결과, 구조화된 JSON, 오디오 — 모두 1급(first-class) 시민.
- **장시간 실행 태스크.** 불투명 수명 주기 + 폴링으로 수 시간짜리 태스크도 간단합니다.

### A2A가 약점을 보이는 분야

- **지연 시간 민감 마이크로 호출.** A2A의 수명 주기는 비동기식. 서브 밀리초 에이전트 간 호출은 적합하지 않음; 직접 RPC 사용.
- **밀접하게 결합된 인프로세스 에이전트.** 두 에이전트가 동일한 Python 프로세스에서 실행된다면, A2A의 HTTP 왕복은 과함.
- **소규모 팀.** 사양 오버헤드가 존재; 내부 전용 에이전트는 형식성이 필요 없을 수 있음.

### A2A vs ACP, ANP, NLIP

2024-2026년에 여러 관련 사양이 등장했습니다:

- **ACP** (IBM/Linux Foundation) — A2A의 전신, 범위가 더 좁음.
- **ANP** (에이전트 네트워크 프로토콜) — 피어 발견 중심, 분산형 우선.
- **NLIP** (Ecma 자연어 상호작용 프로토콜, 2025년 12월 표준화) — 자연어 콘텐츠 유형.

2026년 4월 기준 A2A는 가장 많이 채택된 피어 프로토콜입니다. 비교는 arXiv:2505.02279 (Liu et al., "에이전트 상호운용성 프로토콜 조사")를 참조하세요.

## 구축 방법

`code/main.py`는 `http.server`와 JSON을 사용하는 A2A-최소 서버 및 클라이언트를 구현합니다. 서버는 다음을 수행합니다:

- `/.well-known/agent.json`을 노출,
- `POST /tasks`를 수신,
- 작업 상태 관리,
- `GET /tasks/{id}`에서 아티팩트 반환.

클라이언트는 다음을 수행합니다:

- 에이전트 카드(Agent Card) 가져오기,
- 작업 제출,
- 완료까지 폴링(polling),
- 아티팩트 읽기.

실행 방법:

```
python3 code/main.py
```

스크립트는 백그라운드 스레드에서 서버를 시작한 후 클라이언트를 실행합니다. 전체 흐름(발견, 제출, 폴링, 아티팩트)을 확인할 수 있습니다.

## 사용 방법

`outputs/skill-a2a-integrator.md`는 A2A(애플리케이션 간) 통합을 설계합니다: 에이전트 카드 내용, 작업 스키마, 인증 방식 선택, 스트리밍 대 폴링 방식.

## Ship It

체크리스트:

- **스펙 버전 고정.** A2A는 계속 발전 중이므로, Agent Card에서 프로토콜 버전을 명시해야 합니다.
- **멱등성 있는 작업 생성.** 중복 제출(네트워크 재시도)이 하나의 작업만 생성해야 합니다.
- **아티팩트 스키마.** 에이전트가 반환하는 데이터 형태를 선언해야 하며, 소비자는 이를 검증해야 합니다.
- **속도 제한 + 인증.** A2A는 공개 대상이므로 표준 웹 보안을 적용해야 합니다.
- **실패한 작업의 데드레터 처리.** 시간이 지남에 따라 반복되는 실패 유형을 분석할 수 있도록 해야 합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 클라이언트가 서버를 발견하고 올바른 아티팩트를 수신하는지 확인하세요.
2. 서버에 두 번째 스킬(예: "summarize")을 추가하세요. 에이전트 카드(Agent Card)를 업데이트하세요. 작업 유형에 따라 스킬을 선택하는 클라이언트를 작성하세요.
3. 상태 변경을 방출하는 SSE 스트리밍 엔드포인트 `/tasks/{id}/events`를 구현하세요. 클라이언트는 무엇을 다르게 해야 하나요?
4. A2A 사양(https://a2a-protocol.org/latest/specification/)을 읽으세요. 이 데모가 구현하지 않는 사양이 강제하는 세 가지 사항을 식별하세요.
5. A2A(에이전트 카드 발견)와 MCP(`listTools`를 통한 서버 측 기능 목록)를 비교하세요. 자기 설명형 에이전트와 기능 탐색 간의 트레이드오프는 무엇인가요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| A2A | "에이전트 간(Agent-to-agent)" | 시스템 간 에이전트 호출을 위한 피어 프로토콜. Google 2025. |
| 에이전트 카드(Agent Card) | "에이전트의 명함" | `/.well-known/agent.json`에 위치한 JSON으로 기술, 엔드포인트, 인증 정보를 설명 |
| 태스크(Task) | "작업 단위" | 비동기 상태 객체. 완료 시 아티팩트 생성. 라이프사이클 보유 |
| 아티팩트(Artifact) | "결과물" | 타입이 지정된 출력: 텍스트, 구조화된 JSON, 이미지, 비디오, 오디오. 1급 미디어 |
| 불투명 라이프사이클(Opaque lifecycle) | "해결 방법은 에이전트의 업무" | 클라이언트는 상태 전이만 확인. 서버는 프레임워크/도구 자유롭게 선택 가능 |
| 디스커버리(Discovery) | "에이전트 찾기" | `GET /.well-known/agent.json`이 카드 반환 |
| MCP vs A2A | "도구 vs 피어" | MCP: 수직 에이전트 ↔ 도구. A2A: 수평 에이전트 ↔ 에이전트 |
| ACP / ANP / NLIP | "형제 프로토콜" | 인접 사양. A2A는 2026년 가장 많이 채택된 프로토콜 |

## 추가 자료

- [A2A 사양(A2A specification)](https://a2a-protocol.org/latest/specification/) — 공식 사양 문서
- [Google Developers Blog — A2A 발표](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025년 4월 출시 포스트
- [A2A GitHub 저장소(A2A GitHub repo)](https://github.com/a2aproject/A2A) — 참조 구현 및 SDK
- [Liu et al. — 에이전트 상호 운용성 프로토콜 조사(A Survey of Agent Interoperability Protocols)](https://arxiv.org/html/2505.02279v1) — MCP, ACP, A2A, ANP 비교 분석