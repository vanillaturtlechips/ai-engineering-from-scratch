# 캡스톤 11 — LLM 관측 가능성 및 평가 대시보드

> Langfuse가 오픈코어 모델로 전환했습니다. Arize Phoenix가 2026 GenAI semconv 매핑을 공개했습니다. Helicone과 Braintrust는 사용자별 비용 할당에 집중했습니다. Traceloop의 OpenLLMetry가 사실상의 SDK 계측 도구가 되었습니다. 프로덕션 아키텍처는 ClickHouse(트레이스), Postgres(메타데이터), Next.js(UI), 그리고 샘플링된 트레이스 위에서 실행되는 평가 작업(DeepEval, RAGAS, LLM-judge)의 소규모 군대로 구성됩니다. 자체 호스팅 버전을 구축하고 최소 4개의 SDK 패밀리에서 데이터를 수집하며, 5분 이내에 주입된 회귀 오류를 포착하는 것을 시연하세요.

**유형:** 캡스톤  
**언어:** TypeScript(UI), Python/TypeScript(수집 + 평가), SQL(ClickHouse)  
**선수 조건:** 11단계(LLM 엔지니어링), 13단계(도구), 17단계(인프라), 18단계(안전성)  
**관련 단계:** P11 · P13 · P17 · P18  
**소요 시간:** 25시간

## 문제

2026년에 프로덕션 트래픽을 운영하는 모든 AI 팀은 모델과 함께 관측 가능성 평면(observability plane)을 유지합니다. 비용 할당(cost attribution), 환각(hallucination) 탐지, 드리프트(drift) 모니터링, 탈옥(jailbreak) 신호, SLO 대시보드, PII(개인 식별 정보) 유출 경고 등이 포함됩니다. 오픈소스 참조 도구인 Langfuse, Phoenix, OpenLLMetry는 OpenTelemetry GenAI 시맨틱 컨벤션(semantic conventions)을 수집 스키마로 표준화했습니다. 이제 단일 SDK로 OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM을 계측하고 호환 가능한 스팬(span)을 전송할 수 있습니다.

당신은 최소 4개의 SDK 패밀리에서 데이터를 수집하는 자체 호스팅 대시보드를 구축할 것입니다. 샘플링된 트레이스(trace)에 대해 소규모 평가 작업을 실행하고, 드리프트를 탐지하며, 경고를 발생시켜야 합니다. 측정 기준: 의도적으로 주입된 회귀(프로프트가 PII를 생성하기 시작하는 상황)가 주어졌을 때, 대시보드는 이를 감지하고 5분 이내에 경고를 발령해야 합니다.

## 개념

Ingest는 OTLP HTTP입니다. SDK는 GenAI-semconv 스팬을 생성합니다: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. 스팬은 ClickHouse에 저장되어 컬럼 기반 분석이 수행되며, 메타데이터(사용자, 세션, 앱)는 Postgres에 저장됩니다.

평가는 샘플링된 트레이스 위에서 배치 작업으로 실행됩니다. DeepEval은 충실도(faithfulness), 유해성(toxicity), 답변 관련성(answer relevance)을 점수화합니다. RAGAS는 트레이스에 검색 컨텍스트가 포함된 경우 검색 메트릭을 점수화합니다. 커스텀 LLM-판사는 도메인별 검사(PII 유출, 정책 위반 응답)를 수행합니다. 평가 실행 결과는 부모 트레이스와 연결된 평가 스팬으로 동일한 ClickHouse에 기록됩니다.

드리프트 감지는 시간에 따른 임베딩 공간 분포(프롬프트 임베딩에 대한 PSI 또는 KL 발산)와 평가 점수 추세를 모니터링합니다. 경고는 Prometheus Alertmanager로 전송된 후 Slack/PagerDuty로 전달됩니다. UI는 Recharts를 사용하는 Next.js 15로 구현되었습니다.

## 아키텍처

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  수집기 (수집, 샘플링, 팬아웃)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 아카이브
   (스팬)       (메타데이터)  (원시 이벤트)
       |
       +---> 평가 작업 (DeepEval, RAGAS, LLM-judge)
       |     샘플링 또는 전체 트레이스
       |     평가 스팬 다시 기록
       |
       +---> 드리프트 감지기 (PSI / KL 프롬프트 임베딩)
       |
       +---> Prometheus 메트릭 -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 대시보드 (Recharts)
```

## 스택

- **수집(Ingest)**: OpenTelemetry SDKs + GenAI 시맨틱 컨벤션; OTLP HTTP 전송
- **수집기(Collector)**: 비용 제어를 위한 테일 샘플링 프로세서(tail-sampling processor)가 포함된 OpenTelemetry Collector
- **저장소(Storage)**: 스팬(span) 저장을 위한 ClickHouse, 메타데이터 저장을 위한 Postgres, 원시 이벤트 아카이브를 위한 S3
- **평가(Evals)**: DeepEval, RAGAS 0.2, Arize Phoenix 평가기 팩, 커스텀 LLM-판단기
- **드리프트(Drift)**: 풀링된 프롬프트 임베딩(sentence-transformers)에 대한 PSI / KL 주간 분석
- **경고(Alerting)**: Prometheus Alertmanager → Slack / PagerDuty
- **UI**: Next.js 15 App Router + Recharts + 서버 액션
- **기본 지원 SDK**: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

## 빌드 방법

1. **컬렉터 설정.** OTLP HTTP 수신기, 오류가 발생한 트레이스의 100%와 성공 트레이스의 10%를 유지하는 tail-sampler, ClickHouse 및 S3로의 내보내기를 사용하는 OpenTelemetry Collector를 구성합니다.

2. **ClickHouse 스키마.** GenAI semconv를 미러링하는 열(`gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`)과 긴 페이로드를 위한 JSON 백을 포함하는 테이블 `spans`를 생성합니다. `user_id`와 `app_id`에 대한 보조 인덱스를 추가합니다.

3. **SDK 커버리지 테스트.** OpenLLMetry 자동 계측을 사용하여 각 SDK(OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM)를 사용하는 소규모 클라이언트 앱을 작성합니다. 각각이 ClickHouse에 저장되는 표준 GenAI 스팬을 생성하는지 확인합니다.

4. **평가 작업.** 예약된 작업이 마지막 15분 샘플링된 트레이스를 읽고 DeepEval 충실도, 독성, 답변 관련성을 실행합니다. 출력은 부모 트레이스에 연결된 평가 스팬입니다.

5. **사용자 정의 LLM-판단자.** PII 유출 판단자: 응답을 입력으로 받아 가드 LLM을 호출하여 PII 유출 가능성을 점수화합니다. 높은 점수를 받은 응답은 트리아지 큐에 배치됩니다.

6. **드리프트 감지.** 주간 작업이 이번 주 풀링된 프롬프트 임베딩과 지난 4주 기준선 간의 PSI를 계산합니다. PSI가 임계값을 초과하면 경고를 발생시킵니다.

7. **대시보드.** Next.js 15로 구성된 페이지: 개요(초당 스팬, 사용자당 비용, p95 지연 시간), 트레이스(검색 + 워터폴), 평가(충실도 추세, 독성), 드리프트(시간에 따른 PSI), 경고.

8. **경고 체인.** Prometheus 내보내기는 평가 점수 집계 및 지연 시간 백분위를 읽습니다. Alertmanager는 경고를 Slack으로 라우팅하고 심각한 위반은 PagerDuty로 전송합니다.

9. **회귀 프로브.** 버그 주입: 평가된 챗봇이 1% 확률로 가짜 SSN을 유출하기 시작합니다. MTTR(Mean Time To Repair)을 측정합니다: 버그 배포부터 Slack 경고까지의 시간.

## 사용 방법

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  1개의 트레이스, 3개의 스팬 수신
[clickhouse] 3개의 스팬 삽입 완료 (app=chat, user=u_42)
[eval]       DeepEval 충실도 0.82, 독성 0.03
[drift]      주간 PSI 0.08 (임계값 0.2 미만)
[ui]         라이브 주소: https://obs.example.com
```

## Ship It

`outputs/skill-llm-observability.md`가 결과물입니다. LLM 애플리케이션이 주어지면, 대시보드는 해당 트레이스를 수집하고 평가를 실행하며 드리프트 발생 시 알림을 제공하고 Next.js에서 비용/사용자 분석을 표시합니다.

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 트레이스-스키마 커버리지 | 표준 GenAI 스팬을 생성하는 SDK 패밀리 수 (목표: 6+) |
| 20 | 평가 정확도 | DeepEval / RAGAS 점수 대 수동 레이블링 세트 |
| 20 | 대시보드 UX | 주입된 회귀에 대한 MTTR(평균 복구 시간) (목표: 5분 미만) |
| 20 | 비용 / 확장성 | 백로그 없이 1,000개 스팬/초 지속 수집 |
| 15 | 알림 + 드리프트 감지 | Prometheus/Alertmanager 체인 종단간 테스트 완료 |
| **100** | | |

## 연습 문제

1. Haystack 프레임워크에 대한 커스텀 계측을 추가하세요. 표준 스팬이 `gen_ai.*` 속성과 함께 ClickHouse에 정확히 기록되는지 확인하세요.

2. 동일한 트레이스에서 DeepEval을 Phoenix 평가기로 교체하세요. 두 평가 엔진 간의 점수 드리프트를 측정하세요.

3. 드리프트 감지기 개선: 전역 대신 `app-id`별로 PSI(Population Stability Index)를 계산하세요. 애플리케이션별 드리프트 추이를 표시하세요.

4. "사용자 영향" 페이지 추가: 사용자당 비용(cost-per-user)과 사용자당 실패율(failure-rate-per-user)을 스파크라인과 함께 표시하세요.

5. 테일 샘플링 정책 구현: 독성(toxicity) > 0.5인 트레이스 100% 유지 + 나머지 10% 계층화 샘플링. 도입된 샘플링 편향을 측정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|----------|
| GenAI semconv | "OTel LLM 속성" | LLM 스팬 속성(시스템, 모델, 토큰)에 대한 2025 OpenTelemetry 사양 |
| Tail 샘플링 | "포스트-트레이스 샘플링" | 수집기가 트레이스 완료 후 유지 또는 폐기 결정(오류 확인 가능) |
| PSI | "모집단 안정성 지수" | 두 분포를 비교하는 드리프트 메트릭; 일반적으로 0.2 이상은 의미 있는 드리프트 신호 |
| LLM-심사자 | "모델로서의 평가" | 루브릭(충실도, 유해성, PII)에 따라 다른 LLM 출력을 점수화하는 LLM |
| Tail-샘플링 정책 | "유지 규칙" | 어떤 트레이스를 지속할지 폐기할지 결정하는 규칙; 오류 발생 + 샘플링 비율 |
| 평가 스팬 | "연결된 평가 트레이스" | 원본 LLM 호출 스팬에 연결된 평가 점수를 운반하는 자식 스팬 |
| 사용자당 비용 | "단위 경제학" | 특정 기간 동안 사용자 ID에 할당된 달러 비용; 핵심 제품 지표 |

## 추가 자료

- [Langfuse](https://github.com/langfuse/langfuse) — 참조 오픈코어 관측 가능성 플랫폼
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 강력한 드리프트 지원을 갖춘 대체 참조
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — 자동 계측 SDK 패밀리
- [OpenTelemetry GenAI 시맨틱 컨벤션](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 수집 스키마
- [Helicone](https://www.helicone.ai) — 대체 호스팅 관측 가능성
- [Braintrust](https://www.braintrust.dev) — 대체 평가 우선 플랫폼
- [ClickHouse 문서](https://clickhouse.com/docs) — 컬럼형 스팬 저장소
- [DeepEval](https://github.com/confident-ai/deepeval) — 평가기 라이브러리