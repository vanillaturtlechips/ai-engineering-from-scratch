# OpenTelemetry GenAI — 도구 호출 추적(End-to-End)

> 에이전트가 5개의 도구, 3개의 MCP 서버, 2개의 하위 에이전트를 호출합니다. 이 모든 것을 아우르는 단일 트레이스가 필요합니다. OpenTelemetry GenAI 시맨틱 컨벤션(v1.37 이상 안정 속성)은 2026년 표준이며, Datadog, Langfuse, Arize Phoenix, OpenLLMetry, AgentOps에서 네이티브로 지원됩니다. 이 강의에서는 필요한 속성들을 명명하고, 스팬 계층 구조(에이전트 → LLM → 도구)를 설명하며, 모든 OTel 익스포터와 연동 가능한 stdlib 스팬 이미터를 제공합니다.

**유형:** 구축
**언어:** Python (stdlib, OTel 스팬 이미터)
**사전 요구 사항:** 13단계 · 07 (MCP 서버), 13단계 · 08 (MCP 클라이언트)
**소요 시간:** ~75분

## 학습 목표

- LLM 스팬(span)과 도구 실행(tool-execution) 스팬에 필요한 OTel GenAI 속성(attribute)을 명명할 수 있다.
- 에이전트 루프(agent loop), LLM 호출(LLM call), 도구 호출(tool call), MCP 클라이언트 디스패치(MCP client dispatch)를 포괄하는 트레이스 계층 구조를 구축할 수 있다.
- 캡처할 내용(opt-in)과 삭제할 내용(defaults)을 결정할 수 있다.
- 도구 코드를 재작성하지 않고 로컬 수집기(Jaeger, Langfuse)로 스팬을 전송할 수 있다.

## 문제

2026년 2월의 디버깅 사례: 사용자가 "에이전트가 응답하는 데 30초가 걸릴 때도 있고, 3초 걸릴 때도 있다"고 보고함. 트레이스 없음. 로그에는 LLM 호출은 기록되지만, 툴 디스패치, MCP 서버 왕복, 서브-에이전트 관련 정보는 없음. 추측만 가능. 결국 발견: 하나의 MCP 서버가 콜드 스타트 시 가끔 멈춤 현상 발생.

엔드투엔드 트레이싱이 없으면 이를 찾을 수 없음. OTel GenAI가 해결함.

2025-2026년 OpenTelemetry semantic-conventions 그룹에서 정립된 규약. Datadog, Langfuse, Phoenix, OpenLLMetry, AgentOps 등 모든 백엔드가 동일한 스팬을 파싱할 수 있도록 안정적인 속성 이름을 정의. 한 번 계측; 모든 백엔드로 전송 가능.

## 개념

### 스팬 계층 구조

```
agent.invoke_agent  (최상위, INTERNAL 스팬)
 ├── llm.chat       (CLIENT 스팬)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT 스팬)
 ├── llm.chat       (CLIENT 스팬)
 └── subagent.invoke (INTERNAL)
```

전체 구조는 하나의 트레이스 ID 아래에 중첩됩니다. 스팬 ID는 부모-자식 관계를 연결합니다.

### 필수 속성

2025-2026 semconv에 따라:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — 요청된 모델 문자열 (예: `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — 실제로 제공된 모델.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — 상관관계 추적을 위한 공급자 응답 ID.

도구 스팬의 경우:

- `gen_ai.tool.name` — 도구 식별자.
- `gen_ai.tool.call.id` — 특정 호출 ID.
- `gen_ai.tool.description` — 도구 설명 (선택 사항).

에이전트 스팬의 경우:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### 스팬 종류

- `SpanKind.CLIENT`는 프로세스 경계를 넘는 호출(LLM 공급자, MCP 서버)에 사용됩니다.
- `SpanKind.INTERNAL`은 에이전트의 자체 루프 단계 및 도구 실행에 사용됩니다.

### 콘텐츠 캡처 옵트인

기본적으로 스팬은 메트릭과 타이밍만 포함합니다. 프롬프트나 완료 내용은 포함되지 않습니다. 대용량 페이로드와 PII는 기본적으로 비활성화됩니다. `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 및 특정 콘텐츠 캡처 환경 변수를 설정하여 콘텐츠를 포함할 수 있습니다. 프로덕션 환경에서 활성화하기 전에 주의 깊게 검토하세요.

### 스팬 이벤트

토큰 수준의 이벤트는 스팬 이벤트로 추가할 수 있습니다:

- `gen_ai.content.prompt` — 입력 메시지.
- `gen_ai.content.completion` — 출력 메시지.
- `gen_ai.content.tool_call` — 기록된 도구 호출.

이벤트는 스팬 내에서 시간 순서대로 정렬되어 상세한 재생 기능을 제공합니다.

### 익스포터

OTel 스팬은 다음 플랫폼으로 내보냅니다:

- **Jaeger / Tempo.** 오픈소스, 온프레미스.
- **Langfuse.** LLM 관측 특화; 토큰 사용량 시각화.
- **Arize Phoenix.** 평가 + 트레이싱 통합.
- **Datadog.** 상용; `gen_ai.*` 속성을 기본 파싱.
- **Honeycomb.** 열 기반; 쿼리 친화적.

모두 OTLP(와이어 형식)를 지원합니다. 코드는 신경 쓸 필요 없습니다.

### MCP 간 전파

MCP 클라이언트가 서버를 호출할 때 요청에 W3C `traceparent` 헤더를 삽입합니다. 스트리밍 HTTP는 표준 헤더를 지원합니다. Stdio는 기본적으로 HTTP 헤더를 전달하지 않습니다. 2026 로드맵에서는 JSON-RPC 호출에 `_meta.traceparent` 필드 추가를 논의합니다.

해당 기능이 출시되기 전까지: 모든 요청의 `_meta`에 수동으로 `traceparent`를 포함하세요. 서버는 트레이스 ID를 로깅합니다.

### 메트릭

스팬과 함께 GenAI semconv는 다음 메트릭을 정의합니다:

- `gen_ai.client.token.usage` — 히스토그램.
- `gen_ai.client.operation.duration` — 히스토그램.
- `gen_ai.tool.execution.duration` — 히스토그램.

호출별 세부 정보가 필요하지 않은 대시보드에 사용하세요.

### 에이전트옵스 계층

에이전트옵스(2024년 설립)는 GenAI 관측에 특화되어 있습니다. 인기 있는 프레임워크(LangGraph, Pydantic AI, CrewAI)를 래핑하여 OTel 스팬을 자동으로 생성합니다. 지원되는 프레임워크를 사용하는 스택에 유용하며, 그렇지 않은 경우 수동 계측을 사용하세요.

## 사용 방법

`code/main.py`는 LLM 호출, 두 개의 도구 실행, 하나의 MCP 왕복 작업을 수행하는 에이전트를 위해 OTel 형태의 스팬(span)을 stdout으로 내보냅니다(OTLP-JSON 유사 형식). 실제 내보내기 기능은 없으며, 이 레슨은 스팬 형태와 속성 집합에 중점을 둡니다. 출력 내용을 OTLP 호환 뷰어에 붙여넣거나 직접 읽을 수 있습니다.

확인해야 할 사항:

- 모든 스팬에서 **트레이스 ID(trace id)**가 공유됩니다.
- **부모-자식 관계**는 `parentSpanId`를 통해 인코딩됩니다.
- 필수 **`gen_ai.*` 속성**이 채워집니다.
- 콘텐츠 캡처는 기본적으로 비활성화되어 있으며, 환경 변수를 통해 활성화하는 시나리오가 하나 있습니다.

## Ship It

이 레슨은 `outputs/skill-otel-genai-instrumentation.md`를 생성합니다. 에이전트 코드베이스가 주어졌을 때, 스킬은 계측 계획(instrumentation plan)을 생성합니다: 스팬(span)을 추가할 위치, 채울 속성(attribute), 대상 수출기(exporter) 등.

## 연습 문제

1. `code/main.py`를 실행합니다. 스팬(span) 수를 세고 CLIENT와 INTERNAL이 각각 어떤 것인지 식별하세요.

2. 콘텐츠 캡처(환경 변수)를 활성화하고 `gen_ai.content.prompt` 및 `gen_ai.content.completion` 이벤트가 나타나는지 확인합니다. PII(개인 식별 정보)에 대한 영향을 기록하세요.

3. 도구 실행 메트릭 `gen_ai.tool.execution.duration`을 추가하고 호출당 히스토그램 샘플로 전송하세요.

4. 부모 에이전트 스팬에서 `traceparent`를 MCP 요청의 `_meta.traceparent` 필드로 전파합니다. MCP 서버에서 동일한 트레이스 ID를 확인할 수 있는지 검증하세요.

5. OTel GenAI semconv 사양을 읽습니다. 이 레슨의 코드에서 전송하지 않는 semconv에 나열된 속성 하나를 식별하고 추가하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | 추적(traces), 메트릭(metrics), 로그(logs)를 위한 개방형 표준 |
| GenAI semconv | "GenAI 시맨틱 컨벤션" | LLM / 도구 / 에이전트 스팬(span)을 위한 안정적인 속성 이름 |
| `gen_ai.*` | "속성 네임스페이스" | 모든 GenAI 속성은 이 접두어를 공유함 |
| Span | "시간 측정 작업" | 시작, 종료, 속성을 가진 작업 단위 |
| Trace | "스팬 간 계층 구조" | 추적 ID(trace id)를 공유하는 스팬 트리 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | 스팬 방향에 대한 힌트 |
| OTLP | "OpenTelemetry Line Protocol" | 내보내기용(wire format) 전송 프로토콜 |
| Opt-in content | "프롬프트 / 완료 캡처" | 기본 비활성화; 환경 변수로 활성화 가능 |
| traceparent | "W3C 헤더" | 서비스 간 추적 컨텍스트 전파 |
| Exporter | "백엔드별 전송기" | 스팬을 Jaeger / Datadog 등으로 전송하는 구성 요소 |

## 추가 자료

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI 스팬, 메트릭, 이벤트에 대한 표준 규약
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 및 도구 실행 스팬 속성 목록
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 에이전트 수준의 `invoke_agent` 스팬
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 호스팅 소스
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 프로덕션 통합 가이드