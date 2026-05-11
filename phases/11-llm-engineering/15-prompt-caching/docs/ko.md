# 프롬프트 캐싱과 컨텍스트 캐싱

> 시스템 프롬프트는 4,000토큰입니다. RAG 컨텍스트는 20,000토큰입니다. 모든 요청 시 두 가지를 함께 전송합니다. 또한 매번 비용을 지불합니다. 프롬프트 캐싱을 사용하면 제공업체가 해당 접두사를 서버에 보관해두고 재사용 시 정상 요금의 10%만 청구합니다. 올바르게 사용할 경우 추론 비용을 50–90% 절감하고 첫 토큰 대기 시간을 40–85% 단축할 수 있습니다.

**유형:** 구축(Build)
**언어:** Python
**사전 요구 사항:** 11단계 · 01 (프롬프트 엔지니어링), 11단계 · 05 (컨텍스트 엔지니어링), 11단계 · 11 (캐싱과 비용)
**소요 시간:** ~60분

## 문제

코딩 에이전트가 대화의 모든 턴마다 Claude에게 동일한 15,000토큰 시스템 프롬프트를 전송합니다. 입력 토큰당 $3의 비용으로 20턴을 진행하면 실제 사용자 메시지가 포함되기 전에도 입력 비용만 $0.90가 발생합니다. 이를 하루 10,000건의 대화로 확장하면 변경되지 않는 텍스트에 대해 하루 $9,000의 비용이 청구됩니다.

프롬프트를 축소하면 품질이 저하되므로 불가능합니다. 또한 매 턴마다 프롬프트를 전송하지 않을 수 없습니다 — 모델은 모든 턴에서 이를 필요로 합니다. 유일한 해결책은 공급자가 이미 확인한 접두사에 대해 정가를 지불하지 않는 것입니다.

그 해결책이 바로 프롬프트 캐싱입니다. Anthropic은 2024년 8월에 이를 출시했으며(2025년에는 1시간 확장 TTL 변형 포함), OpenAI는 같은 해 후반에 이를 자동화했고, Google은 Gemini 1.5와 함께 명시적 컨텍스트 캐싱을 출시했습니다. 현재 세 회사 모두 이를 최전방 모델의 1급 기능으로 제공하고 있습니다.

## 개념

![프롬프트 캐싱: 한 번 쓰고 저렴하게 읽기](../assets/prompt-caching.svg)

**메커니즘.** 요청의 접두사가 최근 요청과 일치할 때, 제공자는 토큰을 다시 인코딩하는 대신 이전 실행의 KV-캐시를 제공합니다. 첫 실행 시에는 약간의 쓰기 프리미엄을 지불하고, 이후 실행 시에는 큰 읽기 할인을 받습니다.

**2026년의 세 가지 제공자 유형.**

| 제공자 | API 스타일 | 히트 할인 | 쓰기 프리미엄 | 기본 TTL | 최소 캐시 가능 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 콘텐츠 블록에 명시적 `cache_control` 마커 | 입력 비용 90% 할인 | 25% 추가 요금 | 5분 (최대 1시간 확장 가능) | 1,024 토큰(So넷/Opus), 2,048(Haiku) |
| OpenAI | 자동 접두사 감지 | 입력 비용 50% 할인 | 없음 | 최대 1시간(최적 노력) | 1,024 토큰 |
| Google(Gemini) | 명시적 `CachedContent` API | 저장 비용 청구; 읽기 비용 정가 대비 ~25% | 토큰·시간당 저장 요금 | 사용자 설정(기본 1시간) | 4,096 토큰(Flash), 32,768(Pro) |

**불변 조건.** 세 제공자 모두 접두사만 캐시합니다. 요청 간 토큰이 하나라도 다르면 첫 번째 다른 토큰 이후의 모든 내용은 캐시 미스입니다. *안정적인* 부분은 상단에, *변동적인* 부분은 하단에 배치하세요.

### 캐시 친화적 레이아웃

```
[시스템 프롬프트]          <-- 캐시 대상
[툴 정의]                  <-- 캐시 대상
[소수 샷 예시]             <-- 캐시 대상
[검색된 문서]              <-- 재사용 시 캐시, 아니면 미캐시
[대화 기록]                <-- 마지막 턴까지 캐시
[현재 사용자 메시지]      <-- 절대 캐시하지 않음 (매번 다름)
```

순서를 위반하면 — 사용자 메시지를 시스템 프롬프트 위에 배치하거나, 동적 검색을 소수 샷 사이에 교차시키면 — 캐시가 절대 히트되지 않습니다.

### 손익분기점 계산

Anthropic의 25% 쓰기 프리미엄은 캐시된 블록이 최소 두 번 읽혀야 순비용을 절감할 수 있음을 의미합니다. 1회 쓰기 + 1회 읽기 시 요청당 평균 비용은 0.675x(32% 절감), 1회 쓰기 + 10회 읽기 시 0.205x(80% 절감)입니다. 경험 법칙: TTL 내에서 최소 3회 재사용될 것으로 예상되는 모든 것을 캐시하세요.

## 빌드하기

### 1단계: 명시적 마커가 있는 Anthropic 프롬프트 캐싱

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "당신은 선임 Python 검토자입니다. 평가 기준을 정확히 따르세요.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 마커는 Anthropic에게 5분 동안 블록을 저장하도록 지시합니다. 이 기간 내 재사용은 캐시 히트, 기간 후 재사용은 만료 후 다시 기록됩니다.

**응답 사용량 필드:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # 1.25배 요금
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # 0.1배 요금
```

CI에서 두 필드를 확인하세요 — 요청 간 `cache_read_input_tokens`가 0으로 유지되면 캐시 키가 드리프트되고 있는 것입니다.

### 2단계: 1시간 확장 TTL

장시간 배치 작업의 경우 5분 기본값은 작업 간 만료됩니다. `ttl`을 설정하세요:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1시간 TTL은 쓰기 프리미엄 2배(기본 대비 50% 대신 25%)가 부과되지만, 접두사를 5회 이상 재사용하는 배치 작업에서는 빠르게 비용을 회수합니다.

### 3단계: OpenAI 자동 캐싱

OpenAI는 구성할 항목이 없습니다. 최근 요청과 일치하는 1,024토큰 이상의 접두사는 자동으로 50% 할인을 받습니다.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # 길고 안정적
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # 할인된 부분
```

동일한 캐시 친화적 레이아웃 규칙이 적용됩니다. Anthropic의 캐시를 무효화하지 않는 두 가지 요소(`user` 필드 변경 및 도구 재정렬)는 OpenAI의 캐시를 무효화합니다.

### 4단계: Gemini 명시적 컨텍스트 캐싱

Gemini는 캐시를 생성하고 이름을 지정하는 1급 객체로 취급합니다:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["다음 코드를 검토하세요:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini는 캐시가 유지되는 동안 토큰·시간당 저장 비용을 청구하고, 읽기 비용은 일반 입력 요금의 약 25%입니다. 이는 여러 세션에 걸쳐 동일한 거대한 프롬프트를 며칠 동안 재사용할 때 적합한 구조입니다.

### 5단계: 프로덕션에서 히트율 측정

`code/main.py`에서 쓰기/읽기/미스 횟수를 추적하고 1K 요청당 혼합 비용을 계산하는 시뮬레이션된 3개 공급자 회계 시스템을 참조하세요. 목표 히트율에서 게이트 배포를 제어하세요 — 대부분의 프로덕션 Anthropic 설정은 워밍업 후 80% 이상의 읽기 비율을 보여야 합니다.

## 2026년에도 여전히 발생하는 문제점

- **상단의 동적 타임스탬프.** 시스템 프롬프트 상단에 `"Current time: 2026-04-22 15:30:02"`가 표시됨. 모든 요청이 캐시 미스 발생. 타임스탬프를 캐시 분할 지점 아래로 이동해야 함.
- **도구 재정렬.** 도구를 안정적인 순서로 직렬화 — 배포 간 딕셔너리 재정렬은 모든 캐시 히트를 무효화함.
- **자유 텍스트 유사 중복.** "You are helpful." vs "You are a helpful assistant." — 1바이트 차이로 인해 완전한 캐시 미스 발생.
- **너무 작은 블록 크기.** Anthropic은 1,024토큰(Haiku의 경우 2,048토큰) 최소 크기를 강제함. 더 작은 블록은 캐시되지 않음.
- **맹목적인 비용 대시보드.** "입력 토큰"을 캐시된 것과 캐시되지 않은 것으로 분리하지 않음. 그렇지 않으면 트래픽 감소가 캐시 성공으로 오해될 수 있음.

## 사용 방법

2026년 캐싱 스택:

| 상황 | 선택 |
|-----------|------|
| 안정적인 10k+ 시스템 프롬프트와 많은 턴을 가진 에이전트 | Anthropic `cache_control`과 5분 TTL |
| 30분 이상 접두사를 재사용하는 배치 작업 | `ttl: "1h"`를 사용한 Anthropic |
| GPT-5의 서버리스 엔드포인트, 사용자 정의 인프라 없음 | OpenAI 자동 (접두사를 안정적이고 길게 구성) |
| 거대 코드/문서 코퍼스의 며칠 간 재사용 | Gemini 명시적 `CachedContent` |
| 공급자 간 폴백 | 캐시 가능한 접두사 레이아웃을 공급자 간에 동일하게 유지하여 어떤 히트도 작동하도록 함 |

의미 기반 캐싱(11단계 중 11단계)과 결합하여 사용자-메시지 계층을 처리: 프롬프트 캐싱은 *토큰 동일* 재사용을 처리하고, 의미 기반 캐싱은 *의미 동일* 재사용을 처리합니다.

## Ship It

`outputs/skill-prompt-caching-planner.md` 저장:

```markdown
---
name: prompt-caching-planner
description: 캐시 친화적인 프롬프트 레이아웃을 설계하고 적절한 공급자 캐시 모드를 선택합니다.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

프롬프트(시스템 + 도구 + 몇 가지 예시 + 검색 + 기록 + 사용자)와 사용 프로파일(시간당 요청 수, 필요한 TTL, 공급자)이 주어졌을 때 다음을 출력합니다:

1. 레이아웃. 단일 캐시 중단점이 표시된 재정렬된 섹션; 어떤 섹션이 안정적인지, 어떤 섹션이 휘발성인지 설명합니다.
2. 공급자 모드. Anthropic cache_control, OpenAI automatic, 또는 Gemini CachedContent. TTL과 재사용 패턴을 기반으로 정당화합니다.
3. 손익분기점. TTL 내 쓰기당 예상 읽기 횟수; 캐시 없음 대비 순 비용을 수학적으로 계산합니다.
4. 검증 계획. 두 번째 동일한 요청에서 cache_read_input_tokens > 0임을 확인하는 CI 검증; 캐시된 토큰과 캐시되지 않은 토큰별로 분할된 대시보드.
5. 실패 모드. 이 설정에서 캐시 미스가 발생할 가능성이 가장 높은 세 가지 이유(동적 타임스탬프, 도구 재정렬, 거의 중복된 텍스트)와 각각을 방지하는 방법을 나열합니다.

동적 필드를 중단점 위에 배치하는 캐시 계획은 거부합니다. 재사용 횟수가 2배의 쓰기 프리미엄을 상쇄하지 못하는 1시간 TTL은 활성화하지 않습니다.
```

## 연습 문제

1. **쉬움.** 5,000토큰 시스템 프롬프트를 사용한 10턴 대화를 Claude와 진행하세요. `cache_control` 없이 실행한 후 `cache_control`을 적용하여 실행합니다. 각 경우의 입력 토큰 비용을 보고하세요.
2. **중간.** 프롬프트 템플릿과 요청 로그를 입력으로 받아, 예상 캐시 적중률 및 공급자별(Anthropic 5m, Anthropic 1h, OpenAI 자동, Gemini 명시적) 달러 절감액을 계산하는 테스트 하네스를 작성하세요.
3. **어려움.** 레이아웃 최적화 도구 구축: 프롬프트와 `stable=True/False`로 표시된 필드 목록이 주어졌을 때, 정보를 잃지 않으면서 최대 캐시 친화적 위치에 단일 캐시 중단점을 배치하도록 프롬프트를 재작성하세요. 실제 Anthropic 엔드포인트에서 검증하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 프롬프트 캐싱 | "긴 프롬프트를 저렴하게 만든다" | 일치하는 접두사에 대해 공급자 측 KV-캐시 재사용; 반복 입력 토큰에 대해 50-90% 할인. |
| `cache_control` | "Anthropic 마커" | "여기까지는 캐시 가능"을 선언하는 콘텐츠 블록 속성; `{"type": "ephemeral"}`. |
| 캐시 쓰기 | "프리미엄 지불" | 캐시를 채우는 첫 번째 요청; Anthropic에서는 입력 요금의 ~1.25배, OpenAI에서는 무료. |
| 캐시 읽기 | "할인" | 접두사가 일치하는 후속 요청; Anthropic 10%, OpenAI 50%, Gemini ~25% 요율. |
| TTL | "유지 시간" | 캐시가 유효한 시간(초); Anthropic 기본 5분(최대 1시간 확장 가능), OpenAI 최대 1시간(노력 기반), Gemini 사용자 설정. |
| 확장 TTL | "1시간 Anthropic 캐시" | `{"type": "ephemeral", "ttl": "1h"}`; 쓰기 프리미엄 2배지만 배치 재사용 시 가치 있음. |
| 접두사 일치 | "캐시 미스 원인" | 시작점부터 중단점까지의 모든 토큰이 바이트 단위로 동일할 때만 캐시 적중. |
| 컨텍스트 캐싱(Gemini) | "명시적 캐싱" | Google의 명명된 저장 비용 청구 캐시 객체; 대규모 코퍼스의 다중 일 재사용에 최적.

## 추가 자료

- [Anthropic — 프롬프트 캐싱](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`, 1시간 TTL, 손익분기점 표.
- [OpenAI — 프롬프트 캐싱](https://platform.openai.com/docs/guides/prompt-caching) — 자동 접두사 매칭.
- [Google — 컨텍스트 캐싱](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 및 저장 비용.
- [Anthropic 엔지니어링 — 장문 컨텍스트 워크로드를 위한 프롬프트 캐싱](https://www.anthropic.com/news/prompt-caching) — 지연 시간 수치가 포함된 출시 포스트.
- Phase 11 · 05 (컨텍스트 엔지니어링) — 캐시가 적중할 수 있도록 프롬프트를 분할하는 위치.
- Phase 11 · 11 (캐싱과 비용) — 사용자 메시지에 시맨틱 캐시를 결합한 프롬프트 캐싱.
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — 프롬프트 캐싱이 사용자에게 노출하는 KV-캐시 메모리 모델; 캐시된 접두사를 재계산하는 것보다 재읽는 것이 ~10배 저렴한 이유 설명.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — 프리필은 프롬프트 캐싱이 단축하는 단계; 캐시 적중 시 TTFT(첫 토큰 시간)가 급격히 감소하는 반면 TPOT(전체 처리 시간)는 영향을 받지 않는 이유 설명.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — 프롬프트 캐싱은 추론 비용 곡선을 구부리는 레버로 추측 디코딩, Flash Attention, MQA/GQA와 함께 사용됨; 나머지 세 가지에 대한 설명 포함.