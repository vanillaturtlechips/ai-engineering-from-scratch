# FIPA-ACL과 화행 이론의 유산

> MCP 이전, A2A 이전에 FIPA-ACL이 있었다. 2000년 IEEE 지능형 물리 에이전트 재단(FIPA)은 20개의 수행문(performative), 2개의 콘텐츠 언어, 그리고 계약 네트워크(contract net), 구독/알림(subscribe/notify), 요청-언제(request-when) 등의 상호작용 프로토콜 세트를 갖춘 에이전트 통신 언어를 승인했다. 웹 환경에서는 온톨로지 오버헤드가 너무 커서 산업에서 사라졌지만, LLM 기반 다중 에이전트 시스템의 부활은 형식적 의미론 없이 동일한 아이디어를 조용히 재구현하고 있다: JSON 계약이 수행문을 대체하고, 자연어(natural language)가 온톨로지를 대체한다. 이 레슨은 FIPA-ACL을 진지하게 분석하여 2026년의 어떤 프로토콜 결정이 재창조인지, 어떤 것이 참신성인지, 그리고 현재 물결이 2000년대에 이미 해결한 문제를 어떻게 재발견하게 될지 보여준다.

**유형:** 학습  
**언어:** Python (표준 라이브러리)  
**선수 지식:** 16단계 · 01 (다중 에이전트의 이유)  
**소요 시간:** ~60분

## 문제

2026년 에이전트-프로토콜 환경은 매우 분주합니다: 도구를 위한 MCP, 에이전트를 위한 A2A, 기업 감사를 위한 ACP, 분산 신뢰를 위한 ANP, 자연어 콘텐츠를 위한 NLIP, 그리고 CA-MCP와 24개의 연구 제안이 있습니다. 각 사양은 스스로를 기반 기술(foundational)이라고 선언합니다.

솔직한 평가는 이들 대부분이 매우 구체적인 20년 된 결정 트리를 재발견하고 있다는 것입니다. 오스틴(1962)과 서얼(1969)의 화행 이론(speech-act theory)은 "발화는 행동이다"라는 개념을 제공했습니다. KQML(1993)은 이를 와이어 프로토콜로 변환했습니다. FIPA-ACL(2000년 표준화)은 참조 표준화를 도출했습니다: 20개의 수행적(performatives), 콘텐츠 언어 SL0/SL1, 계약망(contract-net) 및 구독-알림(subscribe-notify)을 위한 상호작용 프로토콜. JADE와 JACK은 자바 참조 플랫폼이었습니다. 이 노력은 2010년경 온톨로지 오버헤드가 너무 무겁고 웹이 승리하면서 사라졌습니다.

MCP의 `tools/call`, A2A의 작업 수명 주기(task lifecycle), 또는 CA-MCP의 공유 컨텍스트 저장소(shared context store)를 볼 때, FIPA 결정의 JSON 네이티브 재해시를 보고 있는 것입니다. 이 유산을 알면 두 가지를 알 수 있습니다: 어떤 새로운 "혁신"이 실제로 재창조된 것인지, 그리고 새로운 사양들이 어떤 오래된 실패 모드를 재발견하게 될지.

## 개념

### 화행 이론, 한 단락으로 요약

오스틴은 일부 문장이 세상을 기술하는 것이 아니라 변화시킨다는 것을 알아냈습니다. "약속합니다." "요청합니다." "선언합니다." 그는 이러한 발화를 수행적 발화(performative utterances)라고 불렀습니다. 설(Searle)은 이를 5가지 범주로 체계화했습니다: 단언적(assertive), 지시적(directive), 약속적(commissive), 표현적(expressive), 선언적(declarative). KQML(Finin et al., 1993)은 소프트웨어 에이전트를 위해 이를 구현했습니다: 메시지는 수행적 발화(행동)와 내용(행동의 대상)으로 구성됩니다. FIPA-ACL은 KQML의 부족한 부분을 정리하고 약 20개의 수행적 발화를 표준화했습니다.

### FIPA 수행적 발화 20가지 (일부 목록)

| 수행적 발화 | 의도 |
|---|---|
| `inform` | "P가 참이라고 알려 드립니다" |
| `request` | "X를 수행해 달라고 요청합니다" |
| `query-if` | "P가 참인가요?" |
| `query-ref` | "X의 값은 무엇인가요?" |
| `propose` | "X를 수행하자고 제안합니다" |
| `accept-proposal` | "제안을 수락합니다" |
| `reject-proposal` | "제안을 거절합니다" |
| `agree` | "X를 수행하기로 동의합니다" |
| `refuse` | "X를 수행하기를 거부합니다" |
| `confirm` | "P가 참임을 확인합니다" |
| `disconfirm` | "P를 부정합니다" |
| `not-understood` | "메시지를 이해하지 못했습니다" |
| `cfp` | "X에 대한 제안 요청" |
| `subscribe` | "X가 변경되면 알려 주세요" |
| `cancel` | "진행 중인 X를 취소합니다" |
| `failure` | "X를 시도했으나 실패했습니다" |

전체 목록은 `fipa00037.pdf`(FIPA ACL 메시지 구조)에 있습니다. 중요한 것은 이를 외우는 것이 아니라, 이 모든 것이 LLM 프로토콜이 결국 다시 추가하는 원시적 요소에 대응한다는 점입니다.

### 표준 FIPA-ACL 메시지

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

7개의 필드는 프로토콜 봉투를 담고, 하나의 필드(`content`)는 페이로드를 담습니다. 나머지 필드는 재시도, 스레딩, 온톨로지를 JSON 프로토콜에 추가할 때마다 매번 재창조하는 것과 정확히 일치합니다.

### 두 가지 레거시 플랫폼

**JADE**(Java Agent DEvelopment 프레임워크, 1999–2020년대)는 가장 많이 사용된 FIPA 호환 런타임이었습니다. 에이전트는 기본 클래스를 확장하고, ACL 메시지를 교환하며, 컨테이너 내에서 실행되고 "행동(behaviors)"을 사용해 조정했습니다. 상호작용 프로토콜 라이브러리에는 계약망(contract-net), 구독-알림(subscribe-notify), 요청-조건(request-when), 제안-수락(propose-accept)이 포함되어 있었습니다.

**JACK**(Agent Oriented Software, 상용)은 FIPA 메시지 위에 BDI(Belief-Desire-Intention) 추론을 강조했습니다. 더 공식적이었지만 덜 채택되었습니다.

두 플랫폼 모두 웹 스택이 다중 에이전트 사용 사례를 흡수하면서 쇠퇴했습니다. MCP와 A2A는 2026년의 런타임 "컨테이너"입니다.

### FIPA가 쇠퇴한 이유

- **온톨로지 오버헤드.** FIPA는 `content`를 파싱하기 위해 공유 온톨로지를 요구했습니다. 온톨로지 합의는 수년에 걸친 표준화 프로세스입니다. 웹은 HTTP + JSON을 사용했습니다.
- **아무도 사용하지 않는 형식적 의미론.** SL(Semantic Language)은 엄격한 진리 조건을 제공했지만, 대부분의 프로덕션 시스템은 자유 형식의 콘텐츠를 사용하고 형식론을 무시했습니다.
- **툴링 종속성.** JADE는 자바 전용이었고, JACK은 상용이었습니다. 다국어 팀은 둘 모두를 우회했습니다.
- **인터넷이 스택에서 승리.** REST, JSON-RPC, gRPC가 ACL의 전송 계층을 대체했습니다.

### LLM 부활은 FIPA-라이트

FIPA `request`와 MCP `tools/call`을 비교:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

같은 봉투, 다른 구문. 둘 다 다음을 포함합니다: 발신자, 수신자, 의도, 페이로드, 상관 ID. 어느 쪽도 다른 쪽보다 혁명적이지 않습니다—동일한 설계에 대한 다른 트레이드오프입니다.

2025년 Liu et al.의 조사("에이전트 상호운용성 프로토콜 조사: MCP, ACP, A2A, ANP", arXiv:2505.02279)는 이 계보를 명확히 합니다: MCP는 도구 사용 화행, A2A는 에이전트-피어 화행, ACP는 감사 추적 화행, ANP는 분산 신원 확장에 대응합니다. 새로운 사양은 JSON 구문과 느슨한 의미론을 가진 ACL의 후손입니다.

### 트레이드오프, 명확하게 설명

**FIPA가 제공했고 현대 사양이 포기한 것:**

- 형식적 의미론 — `inform`이 발신자가 콘텐츠를 믿는다는 것을 증명할 수 있습니다.
- 수행적 발화의 표준 카탈로그 — "취소(`cancel`)"가 필요한지 다시 논쟁할 필요가 없습니다.
- 수십 년간의 상호작용 프로토콜 패턴 — 계약망, 구독-알림, 제안-수락 — 알려진 정확성 속성을 가집니다.

**현대 사양이 제공하고 FIPA가 하지 않은 것:**

- 모든 현대 도구와 호환되는 JSON 네이티브 페이로드.
- 수작업으로 코딩된 온톨로지 없이 LLM이 해석할 수 있는 자연어 콘텐츠.
- 웹 스택 전송 계층(HTTP, SSE, WebSocket).
- 자기 설명 문서를 통한 기능 발견(MCP `listTools`, A2A 에이전트 카드).

더 쉬운 구현을 위한 느슨한 의도 의미론. 이것이 정확한 트레이드오프입니다.

### 이식할 가치가 있는 상호작용 프로토콜

FIPA는 약 15개의 상호작용 프로토콜을 제공했습니다. 3가지는 LLM 다중 에이전트 시스템으로 계승할 가치가 있습니다:

1. **계약망 프로토콜(CNP).** 관리자가 `cfp`(제안 요청)를 발행하고, 입찰자가 `propose`로 응답하며, 관리자가 수락/거절합니다. 이는 표준 작업 시장 패턴(16단계 · 16 협상)입니다.
2. **구독/알림.** 구독자가 `subscribe`를 보내고, 발행자는 주제가 변경될 때마다 `inform`을 보냅니다. 이는 2026년의 모든 이벤트 버스입니다.
3. **요청-조건.** "조건 Y가 충족되면 X를 수행하세요." 사전 조건이 있는 지연 실행. 2026년 유사체는 내구성 있는 워크플로 엔진의 연기된 작업(16단계 · 22 생산 확장)입니다.

각각은 현대 메시지 큐, HTTP + 폴링, 또는 SSE 스트리밍에 깔끔하게 매핑됩니다.

### 온톨로지를 포기할 때 발생하는 문제

공유 온톨로지가 없으면 에이전트는 자연어 콘텐츠에서 의미를 추론합니다. 2026년 문서화된 실패 모드는 **의미적 드리프트**입니다: 두 에이전트가 같은 단어("고객")를 미묘하게 다른 개념으로 사용하고, 수신자 에이전트가 잘못된 해석을 기반으로 행동하며, 스키마 검증기가 이를 포착하지 못합니다. FIPA의 온톨로지 요구사항은 파싱 시점에 메시지를 거부했을 것입니다.

온톨로지를 완전히 도입하지 않는 완화 방법:

- `content`에 JSON 스키마 적용 — 구조적 오류를 와이어에서 거부합니다.
- 타입화된 아티팩트(A2A) — 잘못된 모달리티를 거부합니다.
- 봉투에 명시적 수행적 발화 — 콘텐츠가 자연어일 때도 의도를 명확하게 합니다.

### 2026년 사양, 화행 이론과의 매핑

| 현대 사양 | FIPA 유사체 | 유지하는 것 | 포기하는 것 |
|---|---|---|---|
| MCP `tools/call` | `request` | 명시적 의도, 상관 ID | 형식적 의미론, 온톨로지 |
| MCP `resources/read` | `query-ref` | 명시적 의도, 상관 ID | 형식적 의미론 |
| A2A 작업 수명 주기 | 계약망 + 요청-조건 | 비동기 수명 주기, 상태 전이 | 형식적 완전성 보장 |
| A2A 스트리밍 이벤트 | 구독/알림 | 비동기 푸시 | 타입화된-예측 구독 |
| CA-MCP 공유 컨텍스트 | 블랙보드(Hayes-Roth 1985) | 다중 작성자 공유 메모리 | 논리적 일관성 모델 |
| NLIP | 자연어 콘텐츠 | LLM 네이티브 | 스키마 |

표를 위에서 아래로 읽으면 패턴은 다음과 같습니다: 구조적 원시 요소를 유지하고, 형식론을 포기하며, LLM이 모호성을 메우도록 합니다.

## 빌드하기

`code/main.py`는 순수 표준 라이브러리(pure-stdlib) FIPA-ACL 번역기를 구현합니다. 이 파일은 표준 ACL 봉투(envelope)를 인코딩/디코딩하며, 모든 MCP / A2A 메시지 형식이 동일한 7개 필드로 축소되는 방식을 보여줍니다. 데모 기능:

- MCP 스타일과 A2A 스타일의 메시지 5개를 FIPA-ACL로 인코딩합니다.
- FIPA-ACL을 현대적인 형식으로 다시 디코딩합니다.
- `cfp`, `propose`, `accept-proposal`, `reject-proposal`을 사용하여 관리자 1명과 입찰자 3명 간의 간단한 계약 네트워크(Contract Net) 협상을 실행합니다.

실행 방법:

```
python3 code/main.py
```

출력은 각 현대식 메시지를 2026년 JSON 형식과 FIPA-ACL 형식으로 나란히 표시한 후, 계약 네트워크 입찰(round-trip) 과정을 보여줍니다. 동일한 프로토콜 기본 요소는 변환 과정을 유지되며, 구문만 변경됩니다.

## 사용 방법

`outputs/skill-fipa-mapper.md`는 모든 에이전트 프로토콜 사양을 읽고 FIPA-ACL 매핑을 생성하는 스킬입니다. 새로운 프로토콜을 채택하기 전에 이 스킬을 사용하여 "이것이 진정으로 새로운 것인가, 아니면 JSON 구문을 사용하는 `inform`인가?"라는 질문에 답하세요.  

> **전문 용어 설명**  
> - FIPA-ACL: Foundation for Intelligent Physical Agents - Agent Communication Language  
> - `inform`: FIPA-ACL에서 정보 전달을 위한 기본 수행(performative) 유형

## Ship It

FIPA-ACL을 다시 도입하지 마세요. 대신 다음 체크리스트를 다시 적용하세요:

- 각 메시지의 의도 원시(performative)는 무엇인가?
- 요청-응답 및 취소(cancellation)에 대한 상관 ID가 있는가?
- 명시적 콘텐츠 언어(JSON-RPC, 일반 텍스트, 구조화된 타입 아티팩트)가 있는가?
- 상호작용 프로토콜이 1급(first-class) 객체인가, 아니면 계약망(contract-net)을 처음부터 재구현하고 있는가?
- 두 에이전트가 콘텐츠 의미(semantic drift)에 대해 의견이 다를 때 어떤 일이 발생하는가?

새로운 프로토콜을 프로덕션에 배포하기 전에 이 다섯 가지 질문을 반드시 문서화하세요.

## 연습 문제

1. `code/main.py`를 실행하세요. 왕복 인코딩(round-trip encoding)을 관찰하세요. `tools/call`, `resources/read`, A2A 작업 생성에 대응하는 FIPA 수행어(performative)가 무엇인지 식별하세요.
2. 계약 네트워크(contract-net) 데모에 `cancel` 수행어를 추가하여 관리자가 입찰 중간에 작업을 철회할 수 있도록 확장하세요. `cancel`이 재시도(retries)만으로는 해결할 수 없는 어떤 실패 사례를 해결하나요?
3. FIPA ACL 메시지 구조(http://www.fipa.org/specs/fipa00037/) 4.1–4.3절을 읽으세요. 이 강의에서 다루지 않은 수행어 중 하나를 골라 현대적인 JSON-RPC 유사체를 설명하세요.
4. Liu et al., arXiv:2505.02279를 읽으세요. MCP, A2A, ACP, ANP 각각에 대해 유지하는 FIPA 수행어 패밀리와 제외하는 수행어 패밀리를 나열하세요.
5. 자신의 시스템에서 `request` 수행어의 `content` 필드를 위한 최소한의 JSON 스키마를 설계하세요. 이 스키마가 순수 자연어(natural-language)가 제공하지 않는 것은 무엇이며, 어떤 비용이 발생하나요?

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| 화행(Speech act) | "무언가를 하는 발화" | 오스틴/서얼: 발화를 행동으로 보는 이론. ACL의 이론적 기반. |
| FIPA | "그 오래된 XML 것" | IEEE 지능형 물리 에이전트 재단(Foundation for Intelligent Physical Agents). 2000년 ACL 표준화. |
| ACL | "에이전트 통신 언어(Agent Communication Language)" | FIPA의 봉투 형식: 수행적(performative) + 내용(content) + 메타데이터. |
| 수행적(Performative) | "동사" | 메시지의 의도 클래스: `inform`, `request`, `propose`, `cfp` 등. |
| KQML | "FIPA의 전신" | 지식 쿼리 및 조작 언어(Knowledge Query and Manipulation Language, 1993). 더 단순하고 범위가 좁음. |
| 온톨로지(Ontology) | "공유 어휘" | 내용 언어가 다루는 개념에 대한 형식적 정의. |
| SL0 / SL1 | "FIPA 내용 언어" | 의미 언어(Semantic Language) 레벨 0과 1 — 형식적 내용 언어 패밀리. |
| 계약망(Contract Net) | "작업 시장" | 관리자가 cfp(Call for Proposal) 발행; 입찰자가 제안; 관리자가 수락. 표준 상호작용 프로토콜. |
| 상호작용 프로토콜(Interaction protocol) | "메시지 패턴" | 알려진 정확성을 가진 수행적 시퀀스: request-when, subscribe-notify 등. |

## 추가 자료

- [Liu et al. — 에이전트 상호운용성 프로토콜 조사: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 현대 사양과 FIPA 유산을 연결하는 2025년 표준 조사
- [FIPA ACL 메시지 구조 사양 (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000년 승인된 봉투 형식
- [FIPA 의사소통 행위 라이브러리 사양 (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 전체 수행적(performative) 목록
- [MCP 사양 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — `request`/`query-ref`의 현대적 도구 사용 대응 사양
- [A2A 사양](https://a2a-protocol.org/latest/specification/) — 계약망(contract-net) 및 구독-알림(subscribe-notify)의 현대적 에이전트-피어 대응 사양