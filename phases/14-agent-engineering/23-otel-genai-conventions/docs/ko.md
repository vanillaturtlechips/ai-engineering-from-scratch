# OpenTelemetry GenAI 시맨틱 컨벤션

> OpenTelemetry의 GenAI SIG(2024년 4월 출시)는 에이전트 원격 분석을 위한 표준 스키마를 정의합니다. 스팬 이름, 속성, 콘텐츠 캡처 규칙은 벤더 간에 수렴되어 에이전트 트레이스가 Datadog, Grafana, Jaeger, Honeycomb에서 동일한 의미를 갖도록 합니다.

**유형:** 학습 + 구축  
**언어:** Python (표준 라이브러리)  
**사전 요구 사항:** 14단계 · 13 (LangGraph), 14단계 · 24 (관측 가능성 플랫폼)  
**소요 시간:** ~60분

## 학습 목표

- GenAI 스팬 범주(model/client, agent, tool)의 이름을 말할 수 있다.
- `invoke_agent` CLIENT vs INTERNAL 스팬을 구분하고 각각이 적용되는 경우를 설명할 수 있다.
- 최상위 GenAI 속성(provider name, request model, data-source ID)을 나열할 수 있다.
- 콘텐츠 캡처 계약(content-capture contract)을 설명할 수 있다: 옵트인(opt-in), `OTEL_SEMCONV_STABILITY_OPT_IN`, 외부 참조(external-reference) 권장 사항.

## 문제

모든 벤더는 자체 스팬 이름(span name)을 발명합니다. 운영 팀은 프레임워크별 대시보드를 구축하게 됩니다. OpenTelemetry의 GenAI SIG는 전체 생태계가 대상으로 하는 하나의 표준을 정의함으로써 이 문제를 해결합니다.

## 개념

### 스팬 범주

1. **모델/클라이언트 스팬.** 원시 LLM 호출을 포괄합니다. 프로바이더 SDK(Anthropic, OpenAI, Bedrock) 및 프레임워크 모델 어댑터에서 발행됩니다.
2. **에이전트 스팬.** `create_agent`(에이전트 생성 시)와 `invoke_agent`(실행 시).
3. **툴 스팬.** 툴 호출당 하나씩 생성되며, 부모-자식 관계로 에이전트 스팬과 연결됩니다.

### 에이전트 스팬 명명 규칙

- 스팬 이름: 이름이 지정된 경우 `invoke_agent {gen_ai.agent.name}`; 기본값 `invoke_agent`.
- 스팬 종류:
  - **CLIENT** — 원격 에이전트 서비스용(OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL** — 인프로세스 에이전트 프레임워크용(LangChain, CrewAI, 로컬 ReAct).

### 주요 속성

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — 모델 ID.
- `gen_ai.response.model` — 라우팅으로 인해 요청과 다를 수 있는 해결된 모델.
- `gen_ai.agent.name` — 에이전트 식별자.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — RAG용: 참조된 코퍼스 또는 저장소.

Anthropic, Azure AI Inference, AWS Bedrock, OpenAI에 대한 기술별 규약이 존재합니다.

### 콘텐츠 캡처

기본 규칙: 계측 도구는 기본적으로 입력/출력을 캡처하지 않아야 합니다. 캡처는 다음 옵션을 통해 선택적으로 활성화됩니다:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

권장되는 프로덕션 패턴: 콘텐츠를 외부(S3, 로그 저장소)에 저장하고, 스팬에 참조(텍스트가 아닌 포인터 ID)를 기록합니다. 이는 관측 가능성에 내장된 Lesson 27의 콘텐츠 오염 방어 메커니즘입니다.

### 안정성

2026년 3월 기준으로 대부분의 규약은 실험적 단계입니다. 다음 명령으로 안정화된 프리뷰에 옵트인할 수 있습니다:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+는 GenAI 속성을 기본 LLM 관측 가능성 스키마에 매핑합니다. 다른 백엔드(Grafana, Honeycomb, Jaeger)는 원시 속성을 지원합니다.

### 이 패턴이 실패하는 경우

- **스팬에 전체 프롬프트 캡처.** 운영팀이 읽을 수 있는 트레이스에 PII, 비밀, 고객 데이터 포함. 외부 저장 필요.
- **`gen_ai.provider.name` 누락.** 프로바이더 식별자가 없을 때 멀티프로바이더 대시보드 오작동.
- **부모 링크 없는 스팬.** 고아 툴스팬. 항상 컨텍스트 전파.
- **안정성 옵트인 미설정.** 백엔드 업그레이드 시 속성 이름이 변경될 수 있음.

## 빌드하기

`code/main.py`는 GenAI 규약을 따르는 표준 라이브러리 스팬 발생기를 구현합니다:

- GenAI 속성 스키마를 가진 `Span(스팬)`
- `start_span`과 중첩 컨텍스트를 지원하는 `Tracer(트레이서)`
- `create_agent`, `invoke_agent`(INTERNAL), 도구별 스팬, LLM 호출을 위한 `chat` 스팬을 발생시키는 스크립트 에이전트 실행
- 프롬프트를 외부에 저장하고 스팬에 ID를 기록하는 콘텐츠 캡처 모드

실행 방법:

```
python3 code/main.py
```

출력: 모든 필수 GenAI 속성을 포함한 스팬 트리와 선택적 콘텐츠 참조를 보여주는 "외부 저장소"

## 사용 방법

- **Datadog LLM Observability** (v1.37+)는 속성을 네이티브로 매핑합니다.
- **Langfuse / Phoenix / Opik** (레슨 24) — 생태계 자동 계측.
- **Jaeger / Honeycomb / Grafana Tempo** — 원시 OTel 트레이스; GenAI 속성으로 대시보드 구축.
- **자체 호스팅** — GenAI 프로세서로 OTel Collector를 실행합니다.

## Ship It

`outputs/skill-otel-genai.md`는 OTel GenAI 스팬을 기존 에이전트에 연결하며, 콘텐츠 캡처 기본값과 외부 참조 저장소를 포함합니다.

## 연습 문제

1. `invoke_agent`(INTERNAL) + 도구별 스팬(per-tool spans)을 사용하여 Lesson 01 ReAct 루프를 계측(Instrument)하세요. Jaeger 인스턴스로 전송하세요.
2. "참조만" 모드에서 콘텐츠 캡처를 추가하세요: 프롬프트를 SQLite에 저장하고, 스팬 속성은 행 ID만 전달하도록 구성하세요.
3. `gen_ai.data_source.id` 사양을 읽고 Lesson 09 Mem0 검색에 연결하세요.
4. `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`을 설정하고, 수집기에 의해 속성이 이름이 변경되지 않는지 확인하세요.
5. GenAI 속성만으로 "어떤 도구 오류가 어떤 모델과 상관관계가 있는지" 대시보드를 구축하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI 그룹" | 스키마를 정의하는 OTel 작업 그룹 |
| invoke_agent | "에이전트 스팬" | 에이전트 실행을 나타내는 스팬 이름 |
| CLIENT span | "원격 호출" | 원격 에이전트 서비스 호출에 대한 스팬 |
| INTERNAL span | "인프로세스" | 인프로세스 에이전트 실행에 대한 스팬 |
| gen_ai.provider.name | "프로바이더" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 소스" | 검색 결과가 나온 코퍼스/스토어 |
| Content capture | "프롬프트 로깅" | 메시지 선택적 캡처; 프로덕션 환경에서 외부 저장 |
| Stability opt-in | "프리뷰 모드" | 실험적 규약을 고정하는 환경 변수 |

## 추가 자료

- [OpenTelemetry GenAI 시맨틱 컨벤션](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 사양 문서
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 기본적으로 제공되는 GenAI 스팬
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 내장된 OTel 스팬
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 추적 컨텍스트 전파