# 구조화된 출력 & 제약 디코딩

> LLM에 JSON을 요청하세요. 대부분의 경우 JSON을 얻을 수 있습니다. 하지만 프로덕션 환경에서는 "대부분"이 문제가 됩니다. 제약 디코딩은 샘플링 전 로짓을 편집하여 "대부분"을 "항상"으로 바꿔줍니다.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** Phase 5 · 17 (챗봇), Phase 5 · 19 (서브워드 토크나이징)
**소요 시간:** ~60분

## 문제

분류기가 LLM에 다음과 같이 요청합니다: "다음 중 하나를 반환하세요: {positive, negative, neutral}." 모델은 "이 리뷰의 감정은 긍정적입니다 — 고객이 '...'라고 명시적으로 언급했기 때문에 압도적으로 호의적이기 때문입니다."라고 응답합니다. 파서가 충돌합니다. 분류기의 F1 점수는 0.0입니다.

자유 형식 생성은 계약이 아닙니다. 그것은 제안일 뿐입니다. 프로덕션 시스템에는 계약이 필요합니다.

2026년에는 세 가지 계층이 존재합니다.

1. **프롬프팅.** 공손하게 요청합니다. "JSON 객체만 반환하세요." 프론티어 모델에서는 ~80% 작동하지만, 작은 모델에서는 더 낮은 성능을 보입니다.
2. **네이티브 구조화 출력 API.** OpenAI의 `response_format`, Anthropic의 도구 사용, Gemini JSON 모드. 지원되는 스키마에서 신뢰할 수 있습니다. 벤더 종속적입니다.
3. **제약된 디코딩.** 모든 생성 단계에서 로짓을 수정하여 모델이 *유효하지 않은 토큰을 생성할 수 없도록* 합니다. 구조적으로 100% 유효합니다. 모든 로컬 모델에서 작동합니다.

이 레슨은 세 가지 방법에 대한 직관을 구축하고, 어떤 상황에서 어떤 방법을 사용해야 하는지 설명합니다.

## 개념

![각 단계에서 유효하지 않은 토큰을 마스킹하는 제약 디코딩](../assets/constrained-decoding.svg)

**제약 디코딩 작동 방식.** 각 생성 단계에서 LLM은 전체 어휘(~100k 토큰)에 대한 로짓 벡터를 생성합니다. *로짓 프로세서*는 모델과 샘플러 사이에 위치하며, 현재 타겟 문법(JSON 스키마, 정규식, 문맥 자유 문법) 위치에서 유효한 토큰을 계산하고 유효하지 않은 모든 토큰의 로짓을 음의 무한대로 설정합니다. 남은 로짓에 대한 소프트맥스는 유효한 연속에만 확률 질량을 할당합니다.

2026년 구현 사례:

- **Outlines.** JSON 스키마 또는 정규식을 유한 상태 머신으로 컴파일합니다. 모든 토큰은 O(1) 유효 다음 토큰 조회를 지원합니다. FSM 기반이므로 재귀 스키마는 평탄화가 필요합니다.
- **XGrammar / llguidance.** 문맥 자유 문법 엔진. 재귀적 JSON 스키마를 처리합니다. 디코딩 오버헤드가 거의 없습니다. OpenAI는 2025년 구조화 출력 구현에서 llguidance를 인용했습니다.
- **vLLM 가이드 디코딩.** Outlines, XGrammar 또는 lm-format-enforcer 백엔드를 통해 `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`를 내장 지원합니다.
- **Instructor.** 모든 LLM 위에 Pydantic 기반 래퍼를 제공합니다. 검증 실패 시 재시도합니다. 크로스 프로바이더 호환이지만 로짓을 수정하지 않습니다. 재시도 + 구조화 출력 인식 프롬프트에 의존합니다.

### 반직관적인 결과

제약 디코딩은 종종 *비제약 생성보다 빠릅니다*. 두 가지 이유가 있습니다. 첫째, 다음 토큰 검색 공간을 축소합니다. 둘째, 영리한 구현은 강제 토큰(예: `{"name": "` 같은 스캐폴딩)에 대해 토큰 생성을 완전히 건너뜁니다(모든 바이트가 결정됨).

### 주의해야 할 함정

필드 순서가 중요합니다. `answer`를 `reasoning`보다 먼저 배치하면 모델이 사고하기 전에 답을 확정합니다. JSON은 유효하지만 답이 틀립니다. 검증으로 잡을 수 없습니다.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

스키마 필드 순서는 논리가 아닌 형식입니다.

## 구축 방법

### 1단계: 처음부터 시작하는 정규 표현식 제약 생성

독립형 FSM 구현은 `code/main.py`를 참조하세요. 30줄로 요약한 핵심 아이디어:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM은 지금까지 문법 중 어떤 부분을 만족했는지 추적합니다. `valid_tokens(state, tokenizer)`는 FSM을 수용 경로에서 벗어나지 않고 진전시킬 수 있는 어휘 토큰을 계산합니다.

### 2단계: JSON 스키마 개요

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

검증 오류가 전혀 없습니다. FSM은 잘못된 출력을 도달할 수 없게 만듭니다.

### 3단계: 공급자 중립적 Pydantic을 위한 Instructor

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

다른 메커니즘입니다. Instructor는 로짓을 건드리지 않습니다. 스키마를 프롬프트에 포맷하고, 출력을 파싱하며, 검증 실패 시 재시도(기본 3회)합니다. 모든 공급자와 호환됩니다. 재시도는 지연과 비용을 추가합니다. 공급자 간 이식성이 주요 판매 포인트입니다.

### 4단계: 네이티브 벤더 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

서버 측 제약 디코딩. 지원되는 스키마에 대해 Outlines와 신뢰성 동등. 로컬 모델 관리 불필요. 벤더에 종속됩니다.

## 함정(Pitfalls)

- **재귀적 스키마(Recursive schemas).** 아웃라인은 재귀를 고정된 깊이로 평탄화합니다. 트리 구조 출력(중첩된 댓글, AST 등)은 XGrammar 또는 llguidance(CFG 기반)가 필요합니다.
- **거대한 열거형(Huge enums).** 10,000개 옵션 열거형은 컴파일 속도가 느리거나 타임아웃이 발생할 수 있습니다. 검색기(retriever)로 전환: 먼저 상위 k개 후보를 예측한 후 해당 후보들로 제한합니다.
- **너무 엄격한 문법(Grammar too strict).** `date: "YYYY-MM-DD"` 정규식을 강제하면 모델이 누락된 날짜에 대해 `"unknown"`을 출력할 수 없습니다. 모델은 날짜를 임의로 생성하여 보상합니다. `null` 또는 센티넬(sentinel) 값을 허용하세요.
- **조기 확정(Premature commitment).** 위 필드 순서 함정 참조. 항상 추론(reasoning)을 먼저 배치하세요.
- **스키마 없는 벤더 JSON 모드.** 순수 JSON 모드는 유효한 JSON만 보장할 뿐, *사용 사례에 유효한* JSON을 보장하지 않습니다. 항상 전체 스키마를 제공하세요.

## 사용 방법

2026 스택:

| 상황 | 선택 |
|-----------|------|
| OpenAI/Anthropic/Google 모델, 단순 스키마 | 네이티브 벤더 구조화 출력 |
| 모든 공급자, Pydantic 워크플로우, 재시도 허용 가능 | Instructor |
| 로컬 모델, 100% 유효성 필요, 평탄한 스키마 | Outlines (FSM) |
| 로컬 모델, 재귀적 스키마 | XGrammar 또는 llguidance |
| 자체 호스팅 추론 서버 | vLLM 가이드 디코딩 |
| 재시도 허용 가능한 배치 처리 | Instructor + 가장 저렴한 모델 |

## Ship It

`outputs/skill-structured-output-picker.md`로 저장:

```markdown
---
name: structured-output-picker
description: 구조화된 출력 접근법, 스키마 설계, 검증 계획을 선택합니다.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

사용 사례(공급자, 지연 시간 예산, 스키마 복잡성, 실패 허용도)가 주어졌을 때 다음을 출력합니다:

1. 메커니즘. 네이티브 벤더 구조화 출력, Instructor 재시도, Outlines FSM, XGrammar CFG 중 하나. 한 문장 이유.
2. 스키마 설계. 필드 순서(추론 먼저, 답변 마지막), "알 수 없음"에 대한 nullable 필드, enum 대 regex, 필수 필드.
3. 실패 전략. 최대 재시도 횟수, 대체 모델, 우아한 `null` 처리, 분포 외 거부.
4. 검증 계획. 스키마 준수율(목표 100%), 의미적 유효성(LLM-심사), 필드 커버리지율, 지연 시간 p50/p99.

`answer` 또는 `decision`을 추론 필드보다 앞에 배치하는 설계는 거부합니다. 스키마 없이 bare JSON 모드 사용을 거부합니다. FSM 전용 라이브러리 뒤에 재귀적 스키마를 플래그 처리합니다.
```

## 연습 문제

1. **쉬움.** `Review(sentiment, confidence, evidence_span)`에 대해 제약 디코딩 없이 소형 오픈 웨이트 모델(예: Llama-3.2-3B)을 프롬프트합니다. 100개의 리뷰에서 유효한 JSON으로 파싱되는 비율을 측정합니다.
2. **중간.** Outlines JSON 모드로 동일한 코퍼스를 테스트합니다. 준수율, 지연 시간, 의미적 정확도를 비교합니다.
3. **어려움.** 전화번호(`\d{3}-\d{3}-\d{4}`)에 대해 정규 표현식 제약 디코더를 처음부터 구현합니다. 1000개 샘플에서 유효하지 않은 출력이 0개인지 검증합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 제약 디코딩(Constrained decoding) | 유효한 출력 강제 | 모든 생성 단계에서 유효하지 않은 토큰 로짓(logit)을 마스킹(masking)합니다. |
| 로짓 프로세서(Logit processor) | 제약하는 것 | 함수: `(logits, state) -> masked_logits`. |
| 유한 상태 기계(FSM) | Finite-state machine | 컴파일된 문법 표현; O(1) 유효 다음 토큰 조회. |
| 문맥 자유 문법(CFG) | Context-free grammar | 재귀(recursion)를 처리하는 문법; FSM보다 느리지만 더 표현력이 풍부합니다. |
| 스키마 필드 순서(Schema field order) | 순서가 중요할까? | 예 — 첫 번째 필드가 커밋(commit)됩니다. 항상 답변 전에 추론을 먼저 배치하세요. |
| 유도 디코딩(Guided decoding) | vLLM의 명칭 | 동일한 개념, 추론 서버에 통합되었습니다. |
| JSON 모드(JSON mode) | OpenAI의 초기 버전 | JSON 구문 보장; 스키마 일치는 보장하지 않습니다. |

## 추가 자료

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — 아웃라인 논문.
- [XGrammar 논문 (2024)](https://arxiv.org/abs/2411.15100) — 빠른 CFG 기반 제약 디코딩.
- [vLLM — 구조화된 출력](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 추론 서버 통합.
- [OpenAI — 구조화된 출력 가이드](https://platform.openai.com/docs/guides/structured-outputs) — API 참조 + 주의 사항.
- [Instructor 라이브러리](https://python.useinstructor.com/) — Pydantic + 공급자 간 재시도.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — 6개 제약 디코딩 프레임워크 벤치마킹.