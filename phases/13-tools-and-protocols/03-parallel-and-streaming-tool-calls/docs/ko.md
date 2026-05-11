# 병렬 도구 호출과 스트리밍을 활용한 도구 사용

> 세 개의 독립적인 날씨 조회를 직렬화하면 3번의 왕복이 발생합니다. 병렬로 실행하면 총 시간이 가장 느린 단일 호출 수준으로 축소됩니다. 모든 프론티어 제공업체는 이제 한 번의 턴에서 여러 도구 호출을 발생시킵니다. 그 효과는 분명하지만, 구현은 미묘합니다. 이 강의에서는 두 가지 측면을 모두 다룹니다: 병렬 팬아웃과 스트리밍된 인수 재조립, 특히 ID-상관 관계 함정에 중점을 둡니다.

**유형:** 구축
**언어:** Python (표준 라이브러리, 스레드 풀 + 스트리밍 하네스)
**선수 지식:** 13단계 · 02 (함수 호출 심층 분석)
**소요 시간:** ~75분

## 학습 목표

- `parallel_tool_calls: true`가 존재하는 이유와 비활성화할 시점을 설명.
- 병렬 팬아웃(parallel fan-out) 동안 스트리밍된 인수 청크를 올바른 도구 호출 ID와 연관.
- 조기 파싱 없이 부분 `arguments` 문자열을 완전한 JSON으로 재조립.
- 순차 vs 병렬 지연 시간을 비교하는 3개 도시 날씨 벤치마크 실행.

## 문제

병렬 호출이 없을 때, "벵갈루루, 도쿄, 취리히의 날씨는 어떤가요?"라는 질문에 대한 에이전트의 동작 방식은 다음과 같습니다:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> 실행기 실행, 결과 반환
LLM -> call get_weather(Tokyo)
host -> 실행기 실행, 결과 반환
LLM -> call get_weather(Zurich)
host -> 실행기 실행, 결과 반환
LLM -> 최종 텍스트 답변
```

3번의 LLM 왕복 호출이 발생하며, 각 호출마다 실행기 지연 시간이 추가됩니다. 이는 이상적인 벽시계 시간의 약 4배입니다.

병렬 호출을 사용할 경우:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> 세 실행기를 동시에 실행, 세 결과 반환
LLM -> 최종 텍스트 답변
```

1번의 LLM 왕복 호출만 발생합니다. 실행기 시간은 세 호출 시간의 합이 아닌 최댓값이 됩니다. OpenAI, Anthropic, Gemini에서의 프로덕션 벤치마크 결과, 팬아웃 작업에서 벽시계 시간이 60~70% 감소했습니다.

대신 상관관계 복잡성이라는 비용이 발생합니다. 세 호출이 순서가 뒤바뀌어 완료될 때, 결과에는 매칭을 위한 `tool_call_id`가 포함되어야 모델이 결과를 정렬할 수 있습니다. 결과가 스트리밍될 때는 실행 전에 부분적인 인수 조각을 완전한 JSON으로 조립해야 합니다. Gemini 3은 동일한 도구에 대한 두 병렬 호출이 구분되지 않는 실제 문제를 해결하기 위해 고유 ID를 추가했습니다.

## 개념

### 병렬 실행 활성화

- **OpenAI.** `parallel_tool_calls: true`가 기본값. 직렬 실행을 강제하려면 `false`로 설정.
- **Anthropic.** `disable_parallel_tool_use: false`로 병렬 실행 가능 (Claude 3.5 이상 기본값). 직렬 실행을 원하면 `true`로 설정.
- **Gemini.** 항상 병렬 실행 가능; `tool_config.function_calling_config.mode = "AUTO"`로 설정하면 모델이 자동으로 결정.

도구 간 순서 의존성(`create_file` 후 `write_file`), 한 호출의 출력이 다른 호출의 입력에 영향을 주는 경우, 또는 레이트 리미터가 팬아웃을 처리할 수 없는 경우 병렬 실행을 비활성화.

### ID 상관 관계

모델이 방출하는 모든 호출에는 `id`가 있음. 호스트가 반환하는 모든 결과에는 동일한 `id`가 포함되어야 함. 이 값이 없으면 결과가 모호해짐.

- **OpenAI.** 각 도구 역할 메시지에 `tool_call_id` 포함.
- **Anthropic.** 각 `tool_result` 블록에 `tool_use_id` 포함.
- **Gemini.** 각 `functionResponse`에 `id` 포함 (Gemini 3 이상; Gemini 2는 이름으로 매칭하여 동일 이름의 병렬 호출 시 문제 발생).

### 호출 동시 실행

호스트는 각 호출의 실행기를 별도의 스레드, 코루틴 또는 원격 워커에서 실행. 가장 간단한 구현은 스레드 풀 사용; 프로덕션 환경에서는 `asyncio.gather` 또는 구조화된 동시성을 활용한 비동기 I/O 사용. 완료 순서는 예측할 수 없음 — `id`가 식별자 역할.

흔한 오류: 호출 목록 순서가 아닌 완료 순서로 결과를 회신. 일반적으로 `tool_call_id`만 확인하면 되므로 작동하지만, 결과가 누락되거나 중복될 경우 디버깅이 어려워짐. 명시적 `id`와 함께 완료 순서로 회신하는 것을 권장.

### 스트리밍 도구 호출

모델이 스트리밍할 때 `arguments`는 조각 단위로 도착. 3개의 병렬 호출에 대한 3개의 별도 스트림이 전송 계층에서 인터리브됨. ID별로 하나의 누적기가 필요.

공급자별 형태:

- **OpenAI.** 각 청크는 `choices[0].delta.tool_calls[i].function.arguments` (부분 문자열). 청크에는 `index` (호출 목록 내 위치) 포함. 인덱스별로 누적하고, `id`가 처음 나타날 때 읽으며, `finish_reason = "tool_calls"`일 때 JSON 파싱.
- **Anthropic.** 스트림 이벤트는 `message_start`, 이후 각 블록에 대해 `content_block_start` (유형 `tool_use`, id, 이름, 빈 입력 포함). `content_block_delta` 이벤트는 `input_json_delta` 청크 포함. `content_block_stop`으로 각 블록 종료.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 이상)는 `functionCallId`가 포함된 청크를 방출하여 호출이 깔끔하게 인터리브됨. Gemini 3 이전에는 스트리밍 시 한 번에 하나의 완전한 호출만 반환.

### 부분 JSON 및 조기 파싱 함정

`arguments`가 완성되기 전에는 파싱할 수 없음. `{"city": "Beng`과 같은 부분 JSON은 유효하지 않으며 오류를 발생시킴. 올바른 게이트는 공급자의 호출 종료 신호: OpenAI의 `finish_reason = "tool_calls"`, Anthropic의 `content_block_stop`, 또는 Gemini의 스트림 종료 이벤트. 이때만 `json.loads` 시도. 더 견고한 접근법은 구조가 완성될 때마다 이벤트를 생성하는 증분 JSON 파서 사용; OpenAI 스트리밍 가이드는 실시간 "사고 중" 표시 UX를 위해 이를 권장. 중괄호 카운팅은 완전성 테스트로 신뢰할 수 없음 (따옴표 내부 또는 이스케이프된 콘텐츠의 중괄호가 오탐 유발) — 비공식 디버그 휴리스틱으로만 사용.

### 순서 없는 완료

```
call_A: 빠른 API, 첫 번째로 반환
call_B: 느린 API, 두 번째로 반환
call_C: 중간 속도 API, 세 번째로 반환
```

호스트 회신은 여전히 ID를 명시해야 함:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

회신 순서는 OpenAI 또는 Anthropic의 정확성에 영향을 주지 않음. Gemini는 ID가 일치하는 한 모든 순서를 허용.

### 벤치마크: 순차 vs 병렬

`code/main.py`의 하네스는 400, 600, 800ms 지연 시간을 가진 3개의 실행기를 시뮬레이션. 순차 실행 시 총 1800ms 소요. 병렬 실행 시 max(400, 600, 800) = 800ms 소요. 차이는 상수로, 도구 수에 비례하여 절약 효과 증가.

실제 제약 사항: 병렬 호출은 다운스트림 API에 부하를 줌. 레이트 제한 서비스에 10-way 팬아웃 시 실패 발생. 13단계 · 17단계에서 게이트웨이 수준 백프레셔를 다루며, 재시도 시맨틱스는 향후 단계에서 계획 중.

### 스트리밍 팬아웃 벽시계 시간

모델 자체가 스트리밍하는 경우, 모든 호출이 완료될 때까지 기다리지 않고 하나의 호출 인자가 완성되는 즉시 실행을 시작할 수 있음. 이는 OpenAI가 문서화했으나 모든 SDK에서 노출하지는 않는 최적화 기법. 이 레슨의 하네스는 이를 구현: 시뮬레이션된 스트림이 완전한 인자 객체를 생성하자마자 호스트가 해당 호출을 시작.

## 사용 방법

`code/main.py`는 두 부분으로 구성됩니다. 첫 번째 부분은 `concurrent.futures.ThreadPoolExecutor`를 사용하여 세 번의 시뮬레이션된 날씨 호출을 순차적으로 그리고 병렬로 실행하고 벽시계 시간을 출력합니다. 두 번째 부분은 가짜 스트리밍 응답을 재생합니다 — 세 병렬 호출에 대한 `arguments` 청크가 하나의 스트림에 인터리빙된 상태 — 그리고 `StreamAccumulator`를 사용하여 ID별로 재조립합니다. LLM도 없고 네트워크도 없으며, 오직 재조립 로직만 있습니다.

확인할 사항:

- 순차적 타이머는 1.8초가 소요됩니다. 병렬 타이머는 동일한 가짜 지연 시간에서 0.8초가 소요됩니다.
- 어큐뮬레이터는 ID별로 버퍼링하고 각 호출의 JSON이 완료될 때만 파싱하여 순서가 뒤섞인 청크를 처리합니다.
- 실행기는 모든 스트림이 종료된 후가 아니라 ID의 인수가 확정되는 즉시 작업을 시작합니다.

## Ship It

이 레슨은 `outputs/skill-parallel-call-safety-check.md`를 생성합니다. 도구 레지스트리가 주어졌을 때, 이 스킬은 어떤 도구들이 병렬화해도 안전한지, 어떤 도구들이 순서 의존성을 가지는지, 그리고 어떤 도구들이 다운스트림 속도 제한을 초과할 수 있는지를 감사합니다. 그 결과 각 도구별 `parallel_safe` 플래그가 포함된 수정된 레지스트리를 반환합니다.

## 연습 문제

1. `code/main.py`를 실행하고 시뮬레이션된 지연 시간을 다양하게 변경해 보세요. 병렬-순차 비율이 대략 `max/sum`에 근접하는지 확인하세요 (실제 실행에서는 스레드 스케줄링, 직렬화, 하니스 오버헤드로 인해 이상적인 값과 약간 차이가 발생할 수 있습니다). 어떤 지연 분포에서 병렬화가 더 이상 의미가 없어지나요?

2. "호출이 중간에서 취소됨" 경우를 처리하도록 누적기(accumulator)를 확장하여 버퍼를 삭제하고 `cancelled` 이벤트를 발생시키세요. 어떤 제공업체가 이 경우를 명시적으로 문서화하고 있나요? Anthropic의 `content_block_stop` 시맨틱스와 OpenAI의 `finish_reason: "length"` 동작을 확인해 보세요.

3. 스레드 풀을 `asyncio.gather`로 대체하세요. 두 방식을 벤치마킹하세요. 실제 I/O 작업을 수행하는 실행기(executor)의 경우, 컨텍스트 전환 비용이 낮아 비동기(async) 방식에서 소폭의 성능 향상을 확인할 수 있을 것입니다.

4. 병렬화하지 말아야 할 두 가지 도구(예: `create_file` 후 `write_file`)를 선택하세요. 레지스트리에 `ordering_dependency` 그래프를 추가하고, 이 그래프에 따라 병렬 팬아웃을 제어하세요. 이는 의존성 인식 스케줄링을 위한 최소 기반이며, 향후 에이전트 엔지니어링 단계에서 공식화될 내용입니다.

5. OpenAI의 병렬 함수 호출 섹션과 Anthropic의 `disable_parallel_tool_use` 문서를 읽으세요. Anthropic이 병렬화를 비활성화할 것을 권장하는 실제 도구 유형을 식별하세요. (힌트: 동일한 리소스에 대한 중대한 변경 작업)

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 병렬 도구 호출 | "한 번의 턴에서 팬아웃" | 모델이 단일 어시스턴트 메시지에서 여러 도구 호출을 방출 |
| `parallel_tool_calls` | "OpenAI의 플래그" | 다중 호출 방출 활성화/비활성화 |
| `disable_parallel_tool_use` | "Anthropic의 역방향" | 옵트아웃 플래그; 기본값은 병렬 활성화 |
| 도구 호출 ID | "상관 관계 핸들" | 결과 메시지가 에코해야 하는 호출별 식별자 |
| 누적기 | "스트림 버퍼" | 부분 `arguments` 청크를 위한 ID별 문자열 버퍼 |
| 비순차 완료 | "가장 빠른 것부터" | 병렬 호출이 예측 불가능한 순서로 완료됨; ID가 연결 고리 역할 |
| 종속성 그래프 | "순서 제약 조건" | 다른 도구의 입력에 출력을 제공하는 도구; 병렬화 불가 |
| 조기 파싱 함정 | "JSON.parse 폭발" | 불완전한 `arguments` 문자열 파싱 시도 |
| `streamFunctionCallArguments` | "Gemini 3 기능" | 호출별 고유 ID가 있는 스트리밍 인수 청크 |
| 완료 순서 응답 | "모두 기다리지 않음" | 도착하는 대로 ID를 키로 결과 응답 |

## 추가 자료

- [OpenAI — 병렬 함수 호출](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 기본 동작 및 옵트아웃 플래그
- [Anthropic — 도구 사용: 도구 사용 구현](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` 및 결과 배치 처리
- [Google — Gemini 함수 호출 병렬 섹션](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3의 ID 상관 병렬 호출
- [OpenAI — 도구를 사용한 스트리밍 응답](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 스트림을 위한 청크 기반 인수 재조립
- [Anthropic — 메시지 스트리밍](https://docs.anthropic.com/en/api/messages-streaming) — `input_json_delta`가 포함된 `content_block_delta`