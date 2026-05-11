# 함수 호출 심층 분석 — OpenAI, Anthropic, Gemini

> 2024년 세 선두 제공업체는 동일한 도구 호출 루프에 수렴한 후 나머지 모든 부분에서 분기했습니다. OpenAI는 `tools`와 `tool_calls`를 사용합니다. Anthropic은 `tool_use`와 `tool_result` 블록을 사용합니다. Gemini는 `functionDeclarations`와 고유 ID 상관 관계를 사용합니다. 이 강의에서는 세 제공업체를 나란히 비교하여 한 제공업체에서 작동하는 코드가 다른 제공업체로 이식될 때 깨지지 않도록 합니다.

**유형:** 구축
**언어:** Python (표준 라이브러리, 스키마 변환기)
**선수 조건:** 13단계 · 01 (도구 인터페이스)
**소요 시간:** ~75분

## 학습 목표

- OpenAI, Anthropic, Gemini의 함수 호출 페이로드(선언, 호출, 결과) 간 세 가지 형태 차이를 설명할 수 있다.
- 세 제공업체 형식 간 도구 선언 하나를 번역하고, strict-mode 제약 조건이 어디서 달라질지 예측할 수 있다.
- 각 제공업체에서 `tool_choice`를 사용하여 도구 호출을 강제, 금지 또는 자동 선택할 수 있다.
- 제공업체별 하드 제한(도구 수, 스키마 깊이, 인수 길이)과 제한 위반 시 각 제공업체에서 발생하는 오류 시그니처를 알고 있다.

## 문제

함수 호출 요청의 형태는 공급자마다 다릅니다. 2026년 프로덕션 스택에서 가져온 세 가지 구체적인 예시:

**OpenAI Chat Completions / Responses API.** `tools: [{type: "function", function: {name, description, parameters, strict}}]`를 전달합니다. 모델의 응답에는 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`가 포함되며, 여기서 `arguments`는 파싱해야 하는 JSON 문자열입니다. 엄격 모드(`strict: true`)는 제약된 디코딩을 통해 스키마 준수를 강제합니다.

**Anthropic Messages API.** `tools: [{name, description, input_schema}]`를 전달합니다. 응답은 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 형태로 반환됩니다. `input`은 이미 파싱된 객체(문자열이 아님)입니다. `{type: "tool_result", tool_use_id, content}` 블록을 포함하는 새로운 `user` 메시지로 응답합니다.

**Google Gemini API.** `tools: [{functionDeclarations: [{name, description, parameters}]}]`를 전달합니다(중첩된 `functionDeclarations` 아래에 위치). 응답은 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 형태로 도착하며, 여기서 `id`는 Gemini 3 이상에서 병렬 호출 상관 관계를 위해 고유합니다. `{functionResponse: {name, id, response}}`로 응답합니다.

동일한 루프이지만, 필드 이름이 다르고, 중첩 구조가 다르며, 문자열-객체 관례가 다르며, 상관 관계 메커니즘이 다릅니다. OpenAI에서 날씨 에이전트를 개발한 팀은 Anthropic으로 포팅하는 데 2일, Gemini으로 포팅하는 데 1일을 추가로 소모합니다. 이는 순전히 "배관 작업" 때문입니다.

이 레슨은 세 형식을 하나의 표준 도구 선언으로 통합하고 엣지에서 라우팅하는 번역기를 구축합니다. 13단계 · 17단계는 동일한 패턴을 LLM 게이트웨이로 일반화합니다.

## 개념

### 공통 구조

모든 프로바이더는 다음 5가지가 필요합니다:

1. **도구 목록.** 도구별 이름, 설명, 입력 스키마.
2. **도구 선택.** 특정 도구 강제, 도구 금지, 또는 모델 결정 허용.
3. **호출 발생.** 도구 이름과 인수를 명시하는 구조화된 출력.
4. **호출 ID.** 응답을 올바른 호출과 연관 (병렬 호출 시 중요).
5. **결과 주입.** 결과를 호출에 연결하는 메시지 또는 블록.

### 형태 차이, 필드별 비교

| 측면 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 선언 래퍼 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| 스키마 필드 | `parameters` | `input_schema` | `parameters` |
| 응답 컨테이너 | `tool_calls[]` on assistant message | `content[]` of type `tool_use` | `parts[]` of type `functionCall` |
| 인수 타입 | 문자열화된 JSON | 파싱된 객체 | 파싱된 객체 |
| ID 형식 | `call_...` (OpenAI 생성) | `toolu_...` (Anthropic) | UUID (Gemini 3+) |
| 결과 블록 | role `tool`, `tool_call_id` | `user` with `tool_result`, `tool_use_id` | `functionResponse` with matching `id` |
| 도구 강제 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 도구 금지 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 엄격한 스키마 | `strict: true` | 스키마-계약 (항상 강제) | 요청 레벨의 `responseSchema` |

### 실제로 마주치는 제한

- **OpenAI.** 요청당 128개 도구. 스키마 깊이 5. 인수 문자열 ≤ 8192바이트. 엄격 모드는 `$ref` 금지, `oneOf`/`anyOf`/`allOf` 중복 금지, `required`에 모든 속성 명시 필요.
- **Anthropic.** 요청당 64개 도구. 스키마 깊이는 이론적으로 무제한이지만 실제 한계는 10. 엄격 모드 플래그 없음; 스키마는 계약이며 모델은 일반적으로 준수.
- **Gemini.** 요청당 64개 함수. 스키마 타입은 OpenAPI 3.0 부분집합 (JSON Schema 2020-12와 약간 다름). Gemini 3부터 병렬 호출에 고유 ID 사용.

### `tool_choice` 동작

모든 프로바이더가 지원하는 3가지 모드 (이름은 다름):

- **자동.** 모델이 도구 또는 텍스트 선택. 기본값.
- **필수 / 임의.** 모델은 최소 1개 도구 호출해야 함.
- **없음.** 모델은 도구를 호출하지 않아야 함.

각 프로바이더별 고유 모드:

- **OpenAI.** 이름으로 특정 도구 강제.
- **Anthropic.** 이름으로 특정 도구 강제; `disable_parallel_tool_use` 플래그로 단일/다중 구분.
- **Gemini.** `mode: "VALIDATED"`는 모델 의도와 무관하게 모든 응답을 스키마 검증기로 라우팅.

### 병렬 호출

OpenAI의 `parallel_tool_calls: true` (기본값)는 하나의 어시스턴트 메시지에 여러 호출을 발생. 모든 호출을 실행한 후 `tool_call_id`별 항목을 포함하는 배치 도구 역할 메시지로 응답. Anthropic은 역사적으로 단일 호출만 지원했으나, `disable_parallel_tool_use: false` (Claude 3.5부터 기본값)로 다중 호출 가능. Gemini 2는 병렬 호출을 허용했지만 안정적인 ID를 제공하지 않음; Gemini 3은 UUID를 추가하여 순서가 뒤바뀐 응답도 정확히 연관.

### 스트리밍

세 프로바이더 모두 스트리밍 도구 호출을 지원. 전송 형식은 다음과 같이 다름:

- **OpenAI.** `tool_calls[i].function.arguments`의 델타 청크가 점진적으로 도착. `finish_reason: "tool_calls"`까지 누적.
- **Anthropic.** 블록 시작/델타/종료 이벤트. `input_json_delta` 청크가 부분 인수를 전달.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 신규)는 `functionCallId`가 포함된 청크를 방출하여 여러 병렬 호출이 인터리빙 가능.

Phase 13 · 03은 병렬 + 스트리밍 재조립을 심층 분석. 이 레슨은 선언 및 단일 호출 형태에 집중.

### 오류 및 복구

잘못된 인수 오류도 다르게 나타남:

- **OpenAI (비엄격).** 모델이 `arguments: "{bad json}"` 반환, JSON 파싱 실패, 오류 메시지 주입 후 재호출.
- **OpenAI (엄격).** 디코딩 중 검증 발생; 잘못된 JSON은 불가능하지만 `refusal`이 나타날 수 있음.
- **Anthropic.** `input`에 예상치 못한 필드 포함 가능; 스키마는 참고용. 서버 측에서 검증.
- **Gemini.** OpenAPI 3.0 특이점: 객체 필드의 `enum`은 무시됨; 직접 검증 필요.

### 번역자 패턴

코드에서 표준 도구 선언은 다음과 같습니다 (형태는 선택):

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

세 개의 작은 함수가 이를 각 프로바이더 형태로 변환. `code/main.py`의 하네스는 정확히 이 작업을 수행한 후 각 프로바이더의 응답 형태를 통해 가짜 도구 호출을 왕복. 네트워크 불필요 — 이 레슨은 HTTP가 아닌 형태를 가르침.

프로덕션 팀은 이 번역자를 `AbstractToolset` (Pydantic AI), `UniversalToolNode` (LangGraph), 또는 `BaseTool` (LlamaIndex)로 래핑. Phase 13 · 17은 세 프로바이더 앞에 OpenAI 형태의 API를 노출하는 게이트웨이를 제공.

## 사용 방법

`code/main.py`는 하나의 표준 `Tool` 데이터 클래스와 OpenAI, Anthropic, Gemini 선언 JSON을 생성하는 세 가지 번역기를 정의합니다. 그런 다음 각 형식의 수제 제공자 응답을 동일한 표준 호출 객체로 파싱하여, 표면적 차이에도 불구하고 의미가 동일함을 보여줍니다. 이를 실행하고 세 선언 블록을 나란히 비교하세요.

확인해야 할 사항:

- 세 선언 블록은 봉투(envelope)와 필드 이름만 다릅니다.
- 세 응답 블록은 호출이 위치하는 곳(최상위 `tool_calls`, `content[]` 블록, `parts[]` 항목)에서 차이가 있습니다.
- 하나의 `canonical_call()` 함수가 세 응답 형식 모두에서 `{id, name, args}`를 추출합니다.

## Ship It

이 레슨은 `outputs/skill-provider-portability-audit.md`를 생성합니다. 한 공급자(provider)에 대한 함수 호출(function-calling) 통합이 주어졌을 때, 이 스킬(skill)은 이식성 감사(portability audit)를 생성합니다. 이 감사는 어떤 공급자 제한 사항에 의존하는지, 어떤 필드를 재명명(renaming)해야 하는지, 그리고 다른 공급자로 이식할 때 무엇이 깨지는지(breaks)를 분석합니다.

## 연습 문제

1. `code/main.py`를 실행하고 세 개의 프로바이더 선언 JSON이 모두 동일한 기본 `Tool` 객체를 직렬화하는지 확인하세요. 캐노니컬 도구에 enum 파라미터를 추가하고 OpenAPI 특이점을 처리해야 하는 것은 Gemini 번역기뿐인지 확인하세요.

2. `list_tools` 또는 discovery 호출 후 모델이 반환하는 도구 목록을 추출하는 각 프로바이더용 `ListToolsResponse` 파서를 추가하세요. OpenAI는 네이티브로 지원하지 않습니다. 이 비대칭성을 기록하세요.

3. `tool_choice` 변환 구현: 캐노니컬 `ToolChoice(mode="force", tool_name="x")`를 세 프로바이더 형태로 매핑하세요. 그런 다음 `mode="any"`와 `mode="none"`을 매핑하세요. 레슨의 diff 테이블을 확인하세요.

4. 세 프로바이더 중 하나를 선택하고 함수 호출 가이드를 처음부터 끝까지 읽으세요. 다른 두 프로바이더가 지원하지 않는 스키마 스펙의 필드 하나를 찾으세요. 후보: OpenAI `strict`, Anthropic `disable_parallel_tool_use`, Gemini `function_calling_config.allowed_function_names`.

5. 테스트 벡터 작성: 선언된 스키마를 위반하는 인수를 가진 도구 호출을 생성하세요. 각 프로바이더의 검증기(레슨 01의 stdlib 검증기로 대체 가능)를 통해 실행하고 어떤 오류가 발생하는지 기록하세요. 엄격한 검증이 필요한 프로덕션 환경에서 어떤 프로바이더를 사용할지 문서화하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 함수 호출(Function calling) | "툴 사용(Tool use)" | 구조화된 툴-호출 발신을 위한 프로바이더 수준 API |
| 툴선언(Tool declaration) | "툴 사양(Tool spec)" | 이름 + 설명 + JSON 스키마 입력 페이로드 |
| `tool_choice` | "강제 / 금지(Force / forbid)" | 자동(Auto) / 필수(required) / 없음(none) / 특정 이름(specific-name) 모드 |
| 엄격 모드(Strict mode) | "스키마 강제(Schema enforcement)" | 디코딩을 스키마와 일치하도록 제한하는 OpenAI 플래그 |
| `tool_use` 블록 | "Anthropic의 호출 형태" | id, 이름, 입력을 포함하는 인라인 콘텐츠 블록 |
| `functionCall` 부분 | "Gemini의 호출 형태" | 이름, 인수, id를 포함하는 `parts[]` 항목 |
| 인수-문자열(Arguments-as-string) | "문자열화된 JSON(Stringified JSON)" | OpenAI는 인수를 객체가 아닌 JSON 문자열로 반환 |
| 병렬 툴호출(Parallel tool calls) | "한 번의 턴에서 팬-아웃(Fan-out in one turn)" | 하나의 어시스턴트 메시지에서 여러 툴호출 |
| 거부(Refusal) | "모델이 거절" | 호출 대신 엄격 모드 전용 거부 블록 |
| OpenAPI 3.0 서브셋 | "Gemini 스키마 특이점" | Gemini는 약간의 차이가 있는 JSON-스키마 유사 방언 사용 |

## 추가 자료

- [OpenAI — 함수 호출 가이드](https://platform.openai.com/docs/guides/function-calling) — 엄격 모드(strict mode) 및 병렬 호출(parallel calls) 포함 공식 참조 문서
- [Anthropic — 도구 사용 개요](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 및 `tool_result` 블록 의미론
- [Google — Gemini 함수 호출](https://ai.google.dev/gemini-api/docs/function-calling) — 병렬 호출, 고유 ID, OpenAPI 부분 집합
- [Vertex AI — 함수 호출 참조](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini의 엔터프라이즈 표면
- [OpenAI — 구조화된 출력](https://platform.openai.com/docs/guides/structured-outputs) — 엄격 모드 스키마 강제 적용 세부 정보