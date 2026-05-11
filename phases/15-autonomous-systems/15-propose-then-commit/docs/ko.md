# Human-in-the-Loop: 제안-후-커밋

> 2026년 HITL(Human-in-the-Loop) 합의는 구체적입니다. "에이전트가 질문하고 사용자가 승인 클릭"이 아닙니다. 제안-후-커밋(propose-then-commit) 방식입니다: 제안된 액션은 멱등성 키(idempotency key)와 함께 영구 저장소(durable store)에 기록되며, 검토자에게 의도(intent), 데이터 계보(data lineage), 접근 권한(permissions touched), 영향 범위(blast radius), 롤백 계획(rollback plan)과 함께 표시됩니다. 긍정적 확인(acknowledgement) 후에만 커밋되며, 실행 후 실제 부작용(side effect) 발생 여부를 검증합니다. LangGraph의 `interrupt()`, PostgreSQL 체크포인팅, Microsoft Agent Framework의 `RequestInfoEvent`, Cloudflare의 `waitForApproval()`은 모두 동일한 패턴을 구현합니다. 대표적인 실패 모드는 검토 없이 "승인?"을 클릭하는 고무 도장 승인(rubber-stamp approval)입니다. 문서화된 완화 방법은 명시적 체크리스트를 활용한 질문-응답(challenge-and-response)입니다.

**유형:** 학습(Learn)  
**언어:** Python (표준 라이브러리, 멱등성 기반 제안-후-커밋 상태 머신)  
**선수 지식:** 15단계 · 12 (내구성 있는 실행), 15단계 · 14 (트립와이어)  
**소요 시간:** ~60분

## 문제 정의

에이전트가 액션을 취합니다. 사용자는 이를 승인할지 여부를 결정해야 합니다. 결정이 즉시 이루어진다면, 이는 검토가 아닐 가능성이 높습니다. 결정이 구조화된 경우, 느리지만 신뢰할 수 있습니다. 엔지니어링 관점에서 핵심 질문은 "구조화된 검토를 어떻게 하면 가장 저항이 적은 경로로 만들 수 있는가?"입니다.

2023년식 HITL(Human-in-the-Loop) 패턴은 동기식 프롬프트였습니다: "에이전트가 X에게 본문 Y로 이메일을 보내려 합니다 — 승인하시겠습니까?" 사용자는 승인 버튼을 클릭합니다. 모든 사람이 시스템이 안전하다고 느낍니다. 실제로 이 인터페이스는 과도하게 승인되는 경향이 있습니다: 사용자가 빠르게 승인하고, 승인 기록은 예측력이 낮으며, 에이전트가 오류를 일으킬 경우 감사 로그에는 사용자가 기억하지 못하는 긴 승인 이력이 나타납니다.

2026년식 패턴인 "제안 후 커밋(propose-then-commit)"은 HITL을 지속 가능한 기반 위에 배치하고, 구조화된 메타데이터를 연결하며, 적극적인 커밋을 요구합니다. 모든 관리형 에이전트 SDK에는 이 기능이 포함됩니다: LangGraph의 `interrupt()`, Microsoft Agent Framework의 `RequestInfoEvent`, Cloudflare의 `waitForApproval()`. API 이름은 다르지만, 핵심 구조는 동일합니다.

## 개념

### 제안-후-커밋 상태 머신

1. **제안(Propose).** 에이전트가 제안된 액션을 생성합니다. 영구 저장소(PostgreSQL, Redis, Durable Object)에 저장됩니다. 포함 사항:
   - 의도(intent) (에이전트가 이 작업을 수행하는 이유)
   - 데이터 계보(data lineage) (이 제안으로 이어진 소스)
   - 접근 권한(permissions touched) (어떤 범위/파일/엔드포인트에 접근했는지)
   - 영향 범위(blast radius) (최악의 경우 어떤 일이 발생할 수 있는지)
   - 롤백 계획(rollback plan) (커밋된 경우 어떻게 되돌릴지)
   - 멱등성 키(idempotency key) (제안당 고유 키; 재제출 시 동일한 레코드 반환)
2. **표시(Surface).** 검토자가 모든 메타데이터와 함께 제안을 확인합니다. 검토자는 사람(에이전트 자기 자신이 아님)입니다.
3. **커밋(Commit).** 긍정적 확인. 액션이 실행됩니다.
4. **검증(Verify).** 실행 후 부작용을 다시 읽고 확인합니다. 검증 단계가 실패하면 시스템은 알려진 불량 상태가 되며 경고가 발동됩니다.

### 멱등성 키

멱등성 키가 없으면 일시적 장애 후 재시도 시 승인된 액션이 중복 실행될 수 있습니다. 구체적 예시: 사용자가 "A에서 B로 $100 이체"를 승인했습니다. 네트워크 오류 발생. 워크플로우 재시도. 사용자는 한 번만 승인했지만 이체가 두 번 실행됩니다. 멱등성 키는 승인을 단일 고유 부작용과 연결합니다. 두 번째 실행은 무시(no-op)됩니다.

이는 Stripe와 AWS API에서 사용하는 멱등성 패턴과 동일합니다. Microsoft 에이전트 프레임워크 문서에서 에이전트 승인에 재사용하는 것이 명시되어 있습니다.

### 내구성: 승인 상태가 프로세스보다 오래 지속되는 이유

승인 대기실은 에이전트가 소유하지 않는 상태 조각입니다. 워크플로우는 일시 중지됩니다(레슨 12). 승인이 도착하면 워크플로우는 정확히 그 지점에서 재개됩니다. LangGraph가 `interrupt()`를 PostgreSQL 체크포인팅과 함께 사용하는 이유입니다. 2일 후 승인도 워크플로우를 그대로 찾을 수 있습니다.

### 고무 도장 승인 및 질문-응답 완화 기법

HITL의 기본 UI("승인" / "거부" 버튼)는 진정한 검토 없이 빠른 승인을 생성합니다. 문서화된 완화 기법: "승인" 버튼이 활성화되기 전에 특정 질문에 긍정적 답변을 요구하는 질문-응답 체크리스트. 구체적 형태:

- "이 액션이 어떤 리소스에 접근하는지 이해했나요? [ ]"
- "영향 범위가 허용 가능한지 확인했나요? [ ]"
- "이 액션이 실패할 경우 롤백 계획이 있나요? [ ]"

관료주의를 위한 것이 아닙니다. 강제 기능(forcing function)입니다. 체크박스를 체크할 수 없는 검토자는 설명을 요청(에스컬레이션)하거나 거부(안전한 기본값)합니다. Anthropic의 에이전트 안전 연구는 고무 도장 승인 패턴에 대한 완화 기법으로 체크리스트 기반 HITL을 명시적으로 인용합니다.

### 중대한 액션의 기준

모든 액션에 제안-후-커밋이 필요한 것은 아닙니다. 2026년 지침:

- **중대한 액션**(항상 HITL): 되돌릴 수 없는 쓰기, 금융 거래, 외부 통신, 프로덕션 데이터베이스 변경, 파괴적인 파일 시스템 작업.
- **되돌릴 수 있는 액션**(때때로 HITL): 로컬 파일 편집, 스테이징 환경 변경, 명확한 롤백이 있는 되돌림 가능한 쓰기.
- **읽기 및 검사**(절대 HITL 아님): 파일 읽기, 리소스 목록 조회, 읽기 전용 API 호출.

### 액션 후 검증

"커밋이 실행됨"은 "부작용이 발생함"과 같지 않습니다. 네트워크 분할 및 경쟁 조건으로 인해 백엔드에 지속되지 않은 상태에서 워크플로우가 성공했다고 판단할 수 있습니다. 검증 단계는 커밋 후 대상 리소스를 다시 읽어 확인합니다. 이는 `RETURNING` 절이 있는 데이터베이스 트랜잭션 또는 AWS `PutObject` 후 `GetObject`와 동일한 패턴입니다.

### EU AI 법 제14조

제14조는 EU 내 고위험 AI 시스템에 대해 효과적인 인간 감독을 의무화합니다. "효과적"은 장식적이지 않습니다. 규제 언어는 고무 도장 패턴을 명시적으로 제외합니다. 질문-응답 체크리스트가 포함된 제안-후-커밋은 Microsoft 에이전트 거버넌스 툴킷 준수 문서에서 제14조 검토를 통과하는 형태입니다.

## 사용 방법

`code/main.py`는 표준 라이브러리 Python으로 구현된 제안-후-커밋 상태 머신을 포함합니다. 영구 저장소는 JSON 파일 형식입니다. 멱등성 키는 (thread_id, action_signature)의 해시값입니다. 드라이버는 세 가지 사례를 시뮬레이션합니다: 깨끗한 승인 흐름, 일시적 장애 후 재시도(이중 실행되지 않아야 함), 그리고 기본 승인 흐름 대 도전-응답 흐름 비교입니다.

## Ship It

`outputs/skill-hitl-design.md`는 제안-후-커밋(shape) 작업을 위한 제안된 HITL(Human-in-the-Loop) 워크플로우를 검토하고, 누락된 메타데이터, 멱등성(idempotency), 검증(verification), 또는 도전-응답(challenge-and-response) 레이어를 식별합니다.

## 연습 문제

1. `code/main.py`를 실행합니다. 승인된 제안의 재시도가 영구 기록(durable record)을 사용하고 재실행하지 않는지 확인합니다. 이제 멱등성 키(idempotency key)에 타임스탬프를 포함하도록 변경하고 재시도 시 이중 실행이 발생하는지 보여줍니다.

2. 제안 기록에 `rollback` 필드를 추가합니다. 검증 단계(verify step)에서 실패하는 실행을 시뮬레이션합니다. 롤백이 자동으로 실행되는 것을 보여줍니다.

3. Microsoft Agent Framework의 `RequestInfoEvent` 문서를 읽습니다. API에 포함되어 있지만 토이 엔진(toy engine)에 없는 메타데이터 필드 하나를 식별합니다. 해당 필드를 추가하고 어떤 위험을 방어하는지 설명합니다.

4. 특정 작업(예: "공개 Twitter 계정에 게시")에 대한 도전-응답 체크리스트(challenge-and-response checklist)를 설계합니다. 검토자가 답변해야 하는 세 가지 질문은 무엇이며, 왜 그 세 가지인지 설명합니다.

5. 동기식 "승인하시겠습니까?(Approve?)" 프롬프트만으로 충분한 경우(영구 저장소 필요 없음)를 하나 선택합니다. 그 이유와 함께 수용하는 위험 클래스(risk class)를 명시합니다.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|---|---|---|
| Propose-then-commit | "Two-phase approval" | 영구화된 제안 + 긍정적 커밋 + 검증 |
| Idempotency key | "Retry-safe token" | 제안당 고유값; 두 번째 실행 시 무효(no-ops) |
| Data lineage | "Where it came from" | 제안을 유발한 특정 소스 콘텐츠 |
| Blast radius | "Worst case" | 작업 실패 시 영향 범위 |
| Rubber-stamp | "Fast approval" | 실제 검토 없이 "승인" 클릭 |
| Challenge-and-response | "Forcing checklist" | 검토자가 특정 질문에 반드시 동의해야 함 |
| RequestInfoEvent | "MS Agent Framework primitive" | 구조화된 메타데이터를 가진 지속적 HITL 요청 |
| `interrupt()` / `waitForApproval()` | "Framework primitives" | LangGraph / Cloudflare의 동일한 형태 구현체 |

## 추가 자료

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`, 지속적 승인(durable approvals).
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 및 Durable Objects.
- [Anthropic — 실제 에이전트 자율성 측정](https://www.anthropic.com/research/measuring-agent-autonomy) — 장기 위험 완화를 위한 HITL(Human in the Loop).
- [EU AI Act — 제14조: 인간 감독](https://artificialintelligenceact.eu/article/14/) — 고위험 시스템 규제 기준.
- [Anthropic — Claude의 헌법 (2026년 1월)](https://www.anthropic.com/news/claudes-constitution) — 감독을 위한 헌법적 프레임워크.