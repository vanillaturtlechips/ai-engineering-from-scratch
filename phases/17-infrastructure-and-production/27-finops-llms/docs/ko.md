# LLM을 위한 FinOps — 단위 경제 및 다중 테넌트 속성 분석

> 전통적인 FinOps는 LLM 비용에서 실패합니다. 비용은 토큰-트랜잭션이며, 리소스-가동 시간이 아닙니다. 태그는 매핑되지 않습니다 — API 호출은 트랜잭션이지 자산이 아닙니다. 엔지니어링 결정(프롬프트 설계, 컨텍스트 윈도우, 출력 길이)은 재정적 결정입니다. 2026년 플레이북에는 첫날부터 계측해야 할 세 가지 속성 차원이 있습니다: 좌석 가격 및 확장을 위한 `user_id`별, 제품 표면 비용 및 우선순위 결정을 위한 `task_id` + `route`별, 단위 경제 및 갱신을 위한 `tenant_id`별. 네 가지 토큰 계층 — 프롬프트, 도구, 메모리, 응답 — 하나의 버킷은 비용을 숨깁니다. 다중 테넌트 제품을 위한 강제 계층: 테넌트별 속도 제한(기대 피크의 2-3배, 명확한 429 + 재시도-후); 일일 지출 상한(계약 상한선의 1.5-3배; 속도 제한 강화 + 경고 트리거); 지출 z-점수 > 4에 대한 킬 스위치(자동 일시 중지 + 대기 중 페이지). 속성 패턴: 태그-집계, 텔레메트리-조이너(trace-ID → 청구; 최고 정확도), 샘플링-외삽, 모델 기반 할당, 이벤트-소싱, 실시간 스트리밍. 단위 메트릭: 해결된 쿼리당 비용, 생성된 아티팩트당 비용 — $/M 토큰이 아닙니다. 사후 태깅은 항상 누락됩니다; 요청 생성 시 계측해야 합니다.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 킬 스위치가 있는 장난감 비용-속성 시뮬레이터)  
**선수 조건:** 17단계 · 13 (관측 가능성), 17단계 · 14 (캐싱)  
**소요 시간:** ~60분

## 학습 목표

- 전통적인 FinOps(태그 + 계층)가 LLM 비용에서 실패하는 이유를 설명하고, 세 가지 새로운 어트리뷰션 차원(attribution dimension)을 명명할 수 있다.
- 네 가지 토큰 계층(프롬프트(prompt), 툴(tool), 메모리(memory), 응답(response))을 열거하고, 단일 버킷 과금(single-bucket billing)이 비용을 숨기는 이유를 설명할 수 있다.
- 멀티테넌트 제품(multi-tenant product)을 위한 강제 적용 단계(레이트 제한(rate) → 지출 한도(spend cap) → 킬 스위치(kill switch))를 설계할 수 있다.
- $/M 토큰 대신 단위 메트릭(해결된 쿼리당 비용(cost per resolved query) / 아티팩트당 비용(cost per artifact))을 선택할 수 있다.

## 문제

청구서에 $40,000이 표시되었지만 다음 사항을 알 수 없습니다:
- 어떤 테넌트(tenant)가 이 비용을 발생시켰는지.
- 어떤 제품 기능(product feature)이 이 비용을 유발했는지.
- 개별 사용자(individual user)가 남용(abusive)했는지 여부.
- 프롬프트 블롯(prompt bloat), 툴 호출(tool calls), 또는 메모리 증폭(memory amplification)이 원인인지.

클라우드 리소스(EC2, S3)의 경우 태그(tag)를 통해 집계(aggregate)하는 방식이 작동합니다. 여기서 태그는 라인 아이템(line items)으로 자동 전파됩니다. 그러나 LLM API 호출은 자동 태깅(auto-tag)이 지원되지 않습니다. 호출 지점(call site)에서 사용자(user)/작업(task)/테넌트(tenant) 정보를 명시적으로 스탬프(stamp)하고 이를 전달해야 합니다. 사후 추적(retroactive attribution)은 항상 예외 사례(edge cases)를 놓칩니다.

## 개념

### 세 가지 비용 할당 차원

**사용자별** (`user_id`): 누가 어떤 비용을 발생시키는지. 좌석 가격 책정, 확장 논의, 파워 유저 식별에 활용.

**작업별** (`task_id` + `route`): 어떤 제품 표면이 비용을 발생시키는지. 기능 우선순위 결정, 고비용 기능 중단 결정에 활용.

**테넌트별** (`tenant_id`): 어떤 고객이 수익을 내는지. 단위 경제, 갱신 가격, 티어 임계값 결정에 활용.

첫날부터 세 가지 모두를 호출 사이트에 계측. 사후 적용은 항상 더 나쁨.

### 네 가지 토큰 계층

| 계층 | 예시 | 전체 대비 일반적 비율 |
|-------|---------|---------------------|
| 프롬프트 | 시스템 + 사용자 입력 | 40-60% |
| 도구 | 도구 호출 결과를 피드백 | 20-40% (에이전트 워크로드) |
| 메모리 | 이전 대화 / 검색된 문서 | 10-30% |
| 응답 | 모델 출력 | 10-30% |

네 계층을 함께 묶으면 최적화가 불가능. 비용 할당 스키마에서 분리.

### 강제 적용 단계

1. **테넌트별 속도 제한**: 예상 최대치의 2-3배. `Retry-After`와 함께 429 반환. 테넌트가 마찰 경험; 갑작스러운 청구서 방지.

2. **테넌트별 일일 지출 한도**: 계약 상한선의 1.5-3배. 트리거: 속도 제한 강화 + 고객 성공 팀 알림.

3. **지출 Z-점수 > 4 시 킬 스위치**: 테넌트 기준선 대비. 테넌트 자동 일시 중지; 당직자 호출; 운영 + CS 팀에 에스컬레이션.

### 비용 할당 패턴

- **태그-집계**: 메타데이터 헤더에 스탬프; 나중에 집계. 단순; 대략적.
- **텔레메트리 조인자**: 추적 ID를 통해 청구와 트레이스 결합. 최고 정확도. 성숙한 팀의 방식.
- **샘플링 + 외삽**: 5-10% 샘플링; 곱셈. 대략적 지출에 비용 효율적; 꼬리 부분 누락.
- **모델 기반 할당**: 회귀를 통해 비용 드라이버 추론. 태그가 없는 레거시 데이터용.
- **이벤트 소싱**: 스트림(Kafka / Kinesis) 내 비용 이벤트. 실시간.
- **실시간 스트리밍**: 대시보드 초 단위 업데이트.

### X당 비용이 핵심 지표

$/M 토큰은 벤더 용어. 제품 지표:

- 해결된 지원 티켓당 비용.
- 생성된 기사당 비용.
- 성공적인 에이전트 작업당 비용.
- 사용자 세션 분당 비용.

비용을 제품 결과와 연결. 그렇지 않으면 최적화가 무의미.

### 비용 할당 트레이스 형태

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

모든 호출 시 발행. 데이터 레이크에 저장. 차원별 집계. Phase 17 · 13 관측 가능성 스택에 위치.

### 복합 절감 스택

스택: 캐시 + 배치 + 라우팅 + 게이트웨이. 네 가지 모두 적용 시:
- 캐시 L2 (Phase 17 · 14): 입력 비용 ~10배 절감.
- 배치 (Phase 17 · 15): 50% 할인.
- 저렴한 모델 라우팅 (Phase 17 · 16): 60% 비용 감소.
- 게이트웨이 효율성 (Phase 17 · 19): 중복성 + 재시도.

최적 시나리오: 순진 기준선의 ~5-10%. 대부분의 팀은 2-3개 레버만 사용; 네 가지 모두 적용하는 팀은 드묾.

### 기억해야 할 숫자

- 비용 할당 차원: 사용자별, 작업별, 테넌트별.
- 네 가지 토큰 계층: 프롬프트, 도구, 메모리, 응답.
- 킬 스위치: 지출 Z-점수 > 4.
- 단위 지표: 해결된 쿼리당 비용, $/M 토큰 아님.
- 복합 최적화: 기준선 대비 ~5-10% 가능.

## 사용 방법

`code/main.py`는 3단계 강제 계층 구조를 가진 멀티테넌트 LLM 서비스를 시뮬레이션합니다. 악성 테넌트를 주입하고 킬 스위치 작동 방식을 보여줍니다.

## Ship It

이 레슨은 `outputs/skill-finops-plan.md`를 생성합니다. 제품 및 규모에 따라 어트리뷰션 스키마(attribution schema)와 실행 단계(enforcement ladder)를 설계합니다.

## 연습 문제

1. `code/main.py`를 실행합니다. 킬 스위치(kill switch)는 어떤 z-점수(z-score)에서 발동하나요? 임계값(threshold)은 어떻게 선택하나요?
2. 테넌트별(per-tenant), 작업별(per-task) 비용 대시보드를 설계합니다. 먼저 구축할 5가지 뷰(view)는 무엇인가요?
3. 가장 큰 테넌트(tenant)가 단위 경제(unit-economics)에서 적자(negative)입니다. 고객 영향도(customer impact) 순으로 3가지 개입 방안을 제안하세요.
4. 지원 제품(support product)의 해결된 티켓당 비용(cost per resolved ticket)을 계산합니다: 티켓당 3M 토큰(token), 하루 약 800티켓, GPT-5 캐시(cache) 요금 적용 시.
5. 사후 태깅(retroactive tagging)이 작동할 수 있는지 주장합니다. 언제 사후 태깅이 허용되나요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 사용자별 귀속(Per-user attribution) | "user-level cost" | 모든 호출에 `user_id`가 기록됨 |
| 작업별 귀속(Per-task attribution) | "feature cost" | `task_id` + `route`로 제품 표면 식별 |
| 테넌트별 귀속(Per-tenant attribution) | "customer cost" | `tenant_id`; 단위 경제(unit economics) 결정 |
| 4개 토큰 계층(Four token layers) | "cost layers" | 프롬프트(prompt) + 도구(tool) + 메모리(memory) + 응답(response) |
| 속도 제한(Rate limit) | "429 guard" | 게이트웨이에서 테넌트별 상한선 강제 적용 |
| 일일 지출 한도(Daily spend cap) | "daily ceiling" | 테넌트 범위 예산 + 알림 시스템 |
| 킬 스위치(Kill switch) | "auto-pause" | 지출 z-점수 > 4 시 자동 중단 트리거 |
| 해결당 비용(Cost per resolved) | "product unit metric" | 토큰이 아닌 제품 결과(outcome)에 연결된 비용 |
| 원격 측정 결합기(Telemetry joiner) | "trace-to-billing" | 최고 정확도 귀속 패턴 |
| 중첩 최적화(Stacked optimization) | "cache+batch+route+gateway" | ~5-10% 기본 절감 효과 누적 |

## 추가 자료

- [FinOps Foundation — AI를 위한 FinOps 개요](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — 단위당 비용 2026 가이드](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM 에이전트 비용 할당 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Azure OpenAI의 관리형 LLM](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)