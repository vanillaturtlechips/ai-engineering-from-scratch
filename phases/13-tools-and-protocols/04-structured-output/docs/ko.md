# 구조화된 출력 — JSON 스키마, Pydantic, Zod, 제약 디코딩

> "모델에 JSON 반환을 요청"하는 방식은 최첨단 모델에서도 5~15% 실패율을 보입니다. 구조화된 출력은 제약 디코딩을 통해 이 격차를 해소합니다: 모델이 스키마를 위반하는 토큰을 생성하지 못하도록 물리적으로 차단합니다. OpenAI의 strict 모드, Anthropic의 스키마 타입 도구 사용, Gemini의 `responseSchema`, Pydantic AI의 `output_type`, Zod의 `.parse`는 동일한 개념의 다섯 가지 구현 형태입니다. 이 레슨에서는 학습자들이 모든 프로덕션 추출 파이프라인에 사용할 스키마 검증기와 strict 모드 계약을 구축합니다.

**유형:** 빌드  
**언어:** Python (표준 라이브러리, JSON 스키마 2020-12 서브셋)  
**선수 지식:** 13단계 · 02 (함수 호출 심층 분석)  
**소요 시간:** ~75분

## 학습 목표

- 적절한 제약 조건(enum, min/max, required, pattern)을 사용하여 추출 대상에 대한 JSON Schema 2020-12를 작성합니다.
- 엄격 모드(strict mode)와 제약 디코딩(constrained decoding)이 "생성 후 검증(validate after generation)"과 다른 보장을 제공하는 이유를 설명합니다.
- 세 가지 실패 모드(파싱 오류(parse error), 스키마 위반(schema violation), 모델 거부(model refusal))를 구분합니다.
- 타입 기반 복구(typed repair) 및 타입 기반 거부 처리(typed refusal handling)를 포함한 추출 파이프라인을 구축합니다.

## 문제

구매 주문 이메일을 읽는 에이전트는 자유 텍스트를 `{customer, line_items, total_usd}`로 변환해야 합니다. 세 가지 접근 방식이 있습니다.

**접근 방식 1: JSON 프롬프트.** "customer, line_items, total_usd 필드로 JSON으로 답장해 주세요." 최신 모델에서 85~95% 확률로 작동합니다. 6가지 방식으로 실패합니다: 중괄호 누락, 후행 쉼표, 잘못된 타입, 환각 필드, 토큰 제한으로 인한 잘림, "다음은 JSON입니다:" 같은 유출된 산문.

**접근 방식 2: 생성 후 검증.** 자유롭게 생성한 후 파싱하고 스키마로 검증하며 실패 시 재시도합니다. 신뢰할 수 있지만 비용이 많이 듭니다 — 재시도마다 비용을 지불하며, 잘림 버그는 발생 시마다 한 번의 추가 턴을 소모합니다.

**접근 방식 3: 제약된 디코딩.** 공급자가 디코딩 시 스키마를 강제합니다. 유효하지 않은 토큰은 샘플링 분포에서 마스킹됩니다. 출력은 파싱이 보장되고 검증이 보장됩니다. 실패는 한 가지 모드로 축소됩니다: 거부(모델이 입력이 스키마에 맞지 않는다고 판단).

모든 2026년 최신 공급자는 어떤 형태로든 접근 방식 3을 제공합니다.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}` + 모델이 거부할 경우 응답에 `refusal` 포함.
- **Anthropic.** `tool_use` 입력에 스키마 강제 적용; `stop_reason: "refusal"`은 없지만, 도구 호출 없이 `end_turn`이 신호입니다.
- **Gemini.** 요청 수준에서 `responseSchema`; 2026년 Gemini는 선택된 타입에 대해 토큰 수준 문법 제약 조건을 제공합니다.
- **Pydantic AI.** `output_type=InvoiceModel`은 `InvoiceModel` 타입으로 구조화된 `RunResult`를 반환합니다.
- **Zod (TypeScript).** 공급자의 출력을 Zod 스키마로 검증하는 런타임 파서; OpenAI의 `beta.chat.completions.parse`와 함께 사용.

공통점: 스키마를 한 번 선언하고 종단간으로 강제합니다.

## 개념

### JSON Schema 2020-12 — 공통 언어

모든 공급자는 JSON Schema 2020-12를 지원합니다. 가장 많이 사용하는 구성 요소:

- `type`: `object`, `array`, `string`, `number`, `integer`, `boolean`, `null` 중 하나.
- `properties`: 필드 이름부터 하위 스키마까지의 매핑.
- `required`: 반드시 나타나야 하는 필드 이름 목록.
- `enum`: 허용된 값의 닫힌 집합.
- `minimum` / `maximum` (숫자), `minLength` / `maxLength` / `pattern` (문자열).
- `items`: 모든 배열 요소에 적용되는 하위 스키마.
- `additionalProperties`: `false`는 추가 필드를 금지 (모드에 따라 기본값 다름).

OpenAI 엄격 모드는 세 가지 요구 사항을 추가합니다: 모든 속성은 `required`에 나열되어야 하고, 모든 곳에서 `additionalProperties: false`여야 하며, 해결되지 않은 `$ref`가 없어야 합니다. 이를 위반하면 API가 요청 시 400을 반환합니다.

### Pydantic, Python 바인딩

Pydantic v2는 `model_json_schema()`를 통해 데이터 클래스 형태의 모델에서 JSON 스키마를 생성합니다. Pydantic AI는 이를 래핑하여 다음과 같이 작성할 수 있게 합니다:

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

그리고 에이전트 프레임워크는 스키마를 OpenAI 엄격 모드, Anthropic `input_schema`, 또는 Gemini `responseSchema`로 에지에서 변환합니다. 모델의 출력은 타입이 지정된 `Invoice` 인스턴스로 돌아옵니다. 검증 오류는 타입이 지정된 오류 경로와 함께 `ValidationError`를 발생시킵니다.

### Zod, TypeScript 바인딩

Zod(`z.object({customer: z.string(), ...})`)는 TypeScript의 동등한 도구입니다. OpenAI의 Node SDK는 `zodResponseFormat(Invoice)`를 노출하며, 이는 API의 JSON 스키마 페이로드로 변환됩니다.

### 거부(Refusals)

엄격 모드는 모델이 답변하도록 강제할 수 없습니다. 입력이 스키마에 맞지 않는 경우("이메일이 송장이 아닌 시였음"), 모델은 이유를 포함하는 `refusal` 필드를 방출합니다. 코드는 이를 실패가 아닌 1급 결과로 처리해야 합니다. 거부는 안전 신호로도 유용합니다: 보호된 콘텐츠 이메일에서 신용카드 번호를 추출하라는 요청을 받은 모델은 안전 이유를 첨부한 거부를 반환합니다.

### 오픈 환경에서의 제약된 디코딩

오픈 웨이트 구현은 세 가지 기술을 사용합니다.

1. **문법 기반 디코딩**(`outlines`, `guidance`, `lm-format-enforcer`): 스키마로부터 결정적 유한 상태 기계를 구축; 매 단계에서 FSM을 위반할 토큰의 로짓을 마스킹.
2. **JSON 파서를 이용한 로짓 마스킹**: 모델과 동기화된 스트리밍 JSON 파서 실행; 매 단계에서 유효한 다음 토큰 집합 계산.
3. **검증기를 사용한 추측 디코딩**: 저렴한 초안 모델이 토큰을 제안하고, 검증기가 스키마를 강제.

상용 공급자는 백엔드에서 이 중 하나를 선택합니다. 2026년 최신 기술은 짧은 구조화된 출력에 대해 일반 생성보다 빠르고, 긴 출력에 대해서는 거의 동일한 속도를 보입니다.

### 세 가지 실패 모드

1. **파싱 오류.** 출력이 유효한 JSON이 아님. 엄격 모드에서는 발생할 수 없음. 비엄격 공급자에서는 여전히 발생 가능.
2. **스키마 위반.** 출력은 파싱되지만 스키마를 위반. 엄격 모드에서는 발생할 수 없음. 그 외에는 흔함.
3. **거부.** 모델이 거절. 타입이 지정된 결과로 처리해야 함.

### 재시도 전략

엄격 모드 외부(Anthropic 도구 사용, 비엄격 OpenAI, 이전 Gemini 버전)에서 복구 패턴은 다음과 같습니다:

```
생성 -> 파싱 -> 검증 -> 실패 시 오류 주입 및 재시도, 최대 3회
```

한 번의 재시도로 보통 충분. 세 번의 재시도는 약한 모델의 불안정성을 잡습니다. 세 번 이상은 나쁜 스키마의 신호: 모델이 일부 입력에 대해 스키마를 만족할 수 없으며, 프롬프트나 스키마를 수정해야 함.

### 소형 모델 지원

제약된 디코딩은 소형 모델에서도 작동합니다. 문법 강제 기능이 있는 3B 파라미터 오픈 모델은 구조화 작업에서 원시 프롬프팅을 사용하는 70B 파라미터 모델보다 성능이 우수합니다. 이는 구조화된 출력이 프로덕션에 중요한 주된 이유: 신뢰성을 모델 크기와 분리합니다.

## 사용 방법

`code/main.py`는 표준 라이브러리(stdlib)에 JSON Schema 2020-12 검증기(타입, 필수 필드, enum, 최소/최대, 패턴, 아이템, 추가 속성)를 최소한으로 구현한 파일을 제공합니다. 이 검증기는 `Invoice` 스키마를 래핑하고 가짜 LLM 출력을 검증기를 통해 실행하여 파싱 오류, 스키마 위반, 거부 경로를 보여줍니다. 실제 운영 환경에서는 가짜 출력을 어떤 공급자의 실제 응답으로 교체할 수 있습니다.

주목할 부분:

- 검증기는 경로(path)와 메시지(message)를 포함한 타입화된 `[ValidationError]` 목록을 반환합니다. 이 형태가 재시도 프롬프트에 표시되어야 할 구조입니다.
- 거부 분기는 재시도하지 않습니다. 로깅 후 타입화된 거부 응답을 반환합니다. 14단계 · 09에서는 거부를 안전 신호로 활용합니다.
- `additionalProperties: false` 검사는 적대적 테스트 입력에서 트리거되며, 엄격한 모드가 환각(hallucinated) 필드를 차단하는 이유를 보여줍니다.

## Ship It

이 레슨은 `outputs/skill-structured-output-designer.md`를 생성합니다. 자유 텍스트 추출 대상(인보이스, 지원 티켓, 이력서 등)이 주어졌을 때, 이 스킬은 strict-mode 호환 JSON Schema 2020-12와 이를 미러링하는 Pydantic 모델을 생성하며, 타입화된 거부 및 재시도 처리 스텁이 포함되어 있습니다.

## 연습 문제

1. `code/main.py`를 실행하세요. `total_usd`가 음수인 네 번째 테스트 케이스를 추가하세요. 검증기가 `minimum` 제약 조건 경로로 이를 거부하는지 확인하세요.

2. `oneOf`와 판별자(discriminator)를 지원하도록 검증기를 확장하세요. 일반적인 경우: `line_item`은 `kind`로 태그가 지정된 제품 또는 서비스입니다. 엄격 모드에는 미묘한 규칙이 있습니다. OpenAI의 구조화된 출력 가이드를 확인하세요.

3. 동일한 Invoice 스키마를 Pydantic `BaseModel`로 작성하고 `model_json_schema()` 출력을 직접 작성한 스키마와 비교하세요. Pydantic이 기본적으로 설정하지만 직접 작성한 버전에서는 생략된 하나의 필드를 식별하세요.

4. 거부율을 측정하세요. 추출할 수 없어야 하는 10개의 입력(노래 가사, 수학 증명, 빈 이메일)을 구성하고 엄격 모드로 실제 제공자를 통해 실행하세요. 거부된 경우와 환각(hallucinated)된 출력 수를 세세요. 이는 거부 인식 재시도를 위한 실제 기준입니다.

5. OpenAI의 구조화된 출력 가이드를 처음부터 끝까지 읽으세요. 일반 JSON 스키마에서는 허용되지만 엄격 모드에서 명시적으로 금지하는 하나의 구성 요소를 식별하세요. 그런 다음 금지된 구성 요소를 비본질적으로 사용하는 스키마를 설계하고 엄격 모드와 호환되도록 리팩터링하세요.

## 키 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| JSON Schema 2020-12 | "스키마 스펙" | 모든 최신 공급자가 지원하는 IETF-초안 스키마 방언 |
| Strict 모드 | "보장된 스키마" | 제약된 디코딩을 통해 스키마를 강제하는 OpenAI 플래그 |
| 제약된 디코딩 | "로그잇 마스킹" | 디코딩 시 무효한 다음 토큰을 마스킹하는 실행 시 강제 수단 |
| Refusal | "모델 거부" | 입력이 스키마에 맞지 않을 때의 타입화된 결과 |
| 파싱 오류 | "잘못된 JSON" | JSON으로 파싱되지 않은 출력; Strict 모드에서는 불가능 |
| 스키마 위반 | "잘못된 구조" | 파싱은 되었으나 타입/필수/열거/범위 조건을 위반 |
| `additionalProperties: false` | "추가 속성 불가" | 알 수 없는 필드 금지; OpenAI Strict 모드에서 필수 |
| Pydantic BaseModel | "타입화된 출력" | JSON 스키마를 생성하고 검증하는 파이썬 클래스 |
| Zod 스키마 | "TypeScript 출력 타입" | 공급자 출력 검증을 위한 TS 런타임 스키마 |
| 문법 강제 | "오픈 웨이트 제약 디코딩" | outlines/guidance와 같은 FSM 기반 로그잇 마스킹 |

## 추가 자료

- [OpenAI — 구조화된 출력](https://platform.openai.com/docs/guides/structured-outputs) — 엄격 모드, 거부 응답, 스키마 요구사항
- [OpenAI — 구조화된 출력 소개](https://openai.com/index/introducing-structured-outputs-in-the-api/) — 2024년 8월 출시 발표(디코딩 보장 설명 포함)
- [Pydantic AI — 출력](https://ai.pydantic.dev/output/) — 각 공급자에 직렬화되는 타입 기반 출력_type 바인딩
- [JSON 스키마 — 2020-12 릴리스 노트](https://json-schema.org/draft/2020-12/release-notes) — 표준 사양
- [Microsoft — Azure OpenAI의 구조화된 출력](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) — 엔터프라이즈 배포 참고 사항 및 엄격 모드 주의 사항