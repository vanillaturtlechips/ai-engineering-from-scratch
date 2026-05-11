# 장기 컨텍스트 평가 — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro는 10M 토큰의 컨텍스트를 광고합니다. 1M 토큰에서 8-needle MRCR은 26.3%로 떨어집니다. 광고 ≠ 사용 가능. 장기 컨텍스트 평가는 배포하는 모델의 실제 용량을 알려줍니다.

**유형:** 학습
**언어:** Python
**선수 지식:** Phase 5 · 13 (질문 답변), Phase 5 · 23 (청킹 전략)
**소요 시간:** ~60분

## 문제 정의

200페이지 분량의 계약이 있습니다. 모델은 1M-토큰 컨텍스트를 지원한다고 주장합니다. 계약을 붙여넣고 "해지 조항은 무엇인가요?"라고 질문합니다. 모델은 답변을 생성하지만, 해지 조항이 120k 토큰 위치에 있음에도 불구하고 표지 페이지 내용만을 참조한 답변을 반환합니다. 이는 모델이 실제로 주의를 기울이는 범위를 넘어섰기 때문입니다.

이것이 바로 2026년 컨텍스트 용량 격차입니다. 사양표에는 1M 또는 10M으로 표기되어 있지만, 현실에서는 그 중 60-70%만 사용 가능하며, "사용 가능" 여부는 작업에 따라 달라집니다.

- **검색(단일 정보 탐색):** 최전방 모델들은 광고된 최대치까지 거의 완벽합니다.
- **다중 홉/집계:** 대부분의 모델에서 ~128k를 넘어서면 급격히 성능이 저하됩니다.
- **분산된 사실에 대한 추론:** 가장 먼저 실패하는 작업입니다.

장문 컨텍스트 평가는 이러한 축(axis)들을 측정합니다. 이 강의에서는 벤치마크의 이름, 각 벤치마크가 실제로 측정하는 항목, 그리고 특정 도메인에 맞는 커스텀 니들 테스트(needle test)를 구축하는 방법을 설명합니다.

## 개념

![NIAH 기준선, RULER 다중 작업, LongBench 종합 평가](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** 긴 컨텍스트 내 제어된 깊이에 사실("the magic word is pineapple")을 배치합니다. 모델이 이를 검색하도록 요청합니다. 깊이 × 길이를 스위핑합니다. 최초의 장기 컨텍스트 벤치마크입니다. 현재 프론티어 모델들은 이 평가를 포화시키지만, 이는 필수적이지만 충분하지는 않은 기준선입니다.

**RULER (Nvidia, 2024).** 4개 범주(검색(단일/다중 키/다중 값), 다중 홉 추적(변수 추적), 집계(공통 단어 빈도), QA)에 걸친 13개 작업 유형. 구성 가능한 컨텍스트 길이(4k~128k+). NIAH를 포화시키지만 다중 홉에서 실패하는 모델을 드러냅니다. 2024년 릴리스에서 32k+ 컨텍스트를 주장하는 17개 모델 중 절반만이 32k에서 품질을 유지했습니다.

**LongBench v2 (2024).** 503개 객관식 질문, 8k-2M 단어 컨텍스트, 6개 작업 범주(단일 문서 QA, 다중 문서 QA, 장기 인-컨텍스트 학습, 장기 대화, 코드 저장소, 장기 구조화 데이터). 실제 장기 컨텍스트 동작을 위한 프로덕션 벤치마크입니다.

**MRCR (Multi-Round Coreference Resolution).** 대규모 다중 턴 코레퍼런스. 8-니들, 24-니들, 100-니들 변형. 어텐션 저하 전 모델이 처리할 수 있는 사실 수를 노출합니다.

**NoLiMa.** "비어휘 니들." 니들과 쿼리는 문자적 중복이 없으며, 검색에는 의미적 추론 1단계가 필요합니다. NIAH보다 어렵습니다.

**HELMET.** 많은 문서를 연결(concatenate)하고 그 중 하나의 질문을 합니다. 선택적 어텐션을 테스트합니다.

**BABILong.** bAbI 추론 체인을 관련 없는 헤이스택 내에 임베딩합니다. 단순 검색이 아닌 헤이스택 내 추론을 테스트합니다.

### 실제로 보고할 내용

- **광고된 컨텍스트 창.** 사양 시트의 숫자.
- **유효 검색 길이.** NIAH 통과(일부 임계값, 예: 90%).
- **유효 추론 길이.** 다중 홉 또는 집계 통과(해당 임계값).
- **저하 곡선.** 작업 유형별 컨텍스트 길이 대비 정확도 플롯.

사양 시트를 위한 두 숫자: 검색-유효 및 추론-유효. 일반적으로 추론-유효는 광고된 창의 25-50%입니다.

## 구축 방법

### 1단계: 도메인에 맞는 커스텀 NIAH

`code/main.py`를 참조하세요. 기본 구조:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio는 [0, 1] 범위여야 합니다. 입력값: {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens는 양수여야 합니다. 입력값: {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text에서 토큰이 생성되지 않았습니다")

    # 헤이스택 본문을 채울 수 있을 만큼 필러 반복
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

`depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}를 스윕하세요. 히트맵을 플롯하면 대상 모델의 NIAH 카드가 완성됩니다.

### 2단계: 다중 니들 변형

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

"세 개의 마법 단어는 무엇인가요?"와 같은 질문은 세 니들 모두를 검색해야 합니다. 단일 니들 성공률은 다중 니들 성공률을 예측하지 못합니다.

### 3단계: 멀티홉 변수 추적(RULER 스타일)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "X3은 무엇인가요?"
```

이 답변은 세 할당을 연결해야 합니다. 128k 토큰의 프론티어 모델도 이 작업에서 50-70% 정확도로 떨어지는 경우가 많습니다.

### 4단계: LongBench v2를 스택에 적용

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

카테고리별 정확도를 보고하세요. 집계 점수는 작업 수준의 큰 차이를 숨길 수 있습니다.

## 함정(Pitfalls)

- **NIAH-only 평가.** 1M 토큰에서 NIAH 통과는 멀티홉(multi-hop)과 무관합니다. 항상 RULER 또는 사용자 정의 멀티홉 테스트를 실행하세요.
- **균일 깊이 샘플링.** 많은 구현체에서 depth=0.5만 테스트합니다. depth=0, 0.25, 0.5, 0.75, 1.0을 테스트하세요 — "중간에서 길을 잃는" 효과는 실제로 존재합니다.
- **필러(filler)와의 어휘 중복.** 바늘(needle)이 필러와 키워드를 공유하면 검색이 사소해집니다. NoLiMa 스타일의 비중복 바늘을 사용하세요.
- **지연 시간 무시.** 1M 토큰 프롬프트는 프리필(prefill)에 30-120초가 소요됩니다. 정확도 측정과 함께 첫 토큰 생성 시간(time-to-first-token)을 기록하세요.
- **벤더 자체 보고 수치.** OpenAI, Google, Anthropic은 모두 자체 점수를 공개합니다. 항상 사용 사례에 맞춰 독립적으로 재검증하세요.

## 사용 방법

2026 스택:

| 상황 | 벤치마크 |
|-----------|-----------|
| 빠른 검증 | 3가지 깊이 × 3가지 길이의 커스텀 NIAH |
| 프로덕션용 모델 선택 | 목표 길이에서의 RULER(13개 태스크) |
| 실제 QA 품질 | LongBench v2 single-doc-QA 서브셋 |
| 다중 추론 | BABILong 또는 커스텀 변수 추적 |
| 대화형/대화 | 목표 길이에서의 MRCR 8-needle |
| 모델 업그레이드 회귀 테스트 | 고정 인-하우스 NIAH + RULER 하니스, 모든 새 모델에 실행 |

프로덕션용 경험 법칙: 의도한 길이에서 NIAH + 1 추론 태스크를 통과하기 전까지는 어떤 컨텍스트 윈도우도 신뢰하지 마세요.

## Ship It

`outputs/skill-long-context-eval.md`로 저장:

```markdown
---
name: long-context-eval
description: 주어진 모델 및 사용 사례에 대한 장문 컨텍스트 평가 체계를 설계합니다.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

대상 모델, 목표 컨텍스트 길이, 사용 사례를 고려하여 다음을 출력합니다:

1. 테스트. NIAH(Needle In A Haystack) 깊이 × 길이 그리드; RULER 멀티스텝; 커스텀 도메인 태스크.
2. 샘플링. 각 길이에서 깊이 0, 0.25, 0.5, 0.75, 1.0.
3. 메트릭. 검색 성공률; 추론 성공률; 첫 토큰 생성 시간; 쿼리당 비용.
4. 커트오프. 유효 검색 길이(90% 성공률) 및 유효 추론 길이(70% 성공률). 둘 다 보고.
5. 회귀. 고정 하니스, 모든 모델 업그레이드 시 재실행, 변화량 표시.

모델 카드의 컨텍스트 윈도우만 신뢰하지 마십시오. 멀티스텝 작업 부하에 대해 NIAH 전용 평가를 거부하십시오. 공급업체 자체 보고 장문 컨텍스트 점수를 독립적 증거로 사용하지 마십시오.
```

## 연습 문제

1. **쉬움.** 3개의 깊이(0.25, 0.5, 0.75) × 3개의 길이(1k, 4k, 16k)로 NIAH(Needle-in-a-Haystack)를 구성합니다. 임의의 모델에서 실행하고 통과율을 3×3 히트맵으로 시각화합니다.  
2. **중간.** 3-needle 변형 버전을 추가합니다. 각 길이에서 3개의 니들 검색 성능을 측정합니다. 동일한 길이에서의 단일 니들 통과율과 비교합니다.  
3. **어려움.** 64k의 필러 데이터 안에 3홉(X1 → X2 → X3) 구조의 변수 추적 과제를 설계합니다. 3개의 최신 모델(Frontier Models)에서 정확도를 측정하고, 모델별 유효 추론 길이(Effective Reasoning Length)를 보고합니다.  

> **참고:**  
> - NIAH(Needle-in-a-Haystack): "건초 더미 속 바늘" 테스트  
> - Frontier Models: 최신 고성능 모델 (예: GPT-4, Claude 3 등)  
> - 유효 추론 길이: 모델이 문맥 의존성을 유지할 수 있는 최대 토큰 길이

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| NIAH | 건초 더미 속 바늘 | 사실을 채움(filler) 속에 심어 놓고, 모델이 이를 검색하도록 요청. |
| RULER | 스테로이드 NIAH | 검색 / 다중 홉 / 집계 / QA(질의응답) 분야 13가지 작업 유형. |
| 유효 컨텍스트 | 실제 용량 | 정확도가 여전히 임계값 이상으로 유지되는 길이. |
| 중간에서 길을 잃음 | 깊이 편향 | 모델이 긴 입력의 중간 내용에는 주의를 덜 기울임. |
| 다중 바늘 | 한 번에 많은 사실 | 여러 사실 심기; 검색만이 아닌 주의 분산(attention juggling) 테스트. |
| MRCR | 다중 라운드 코퍼퍼런스 | 8, 24, 또는 100-바늘 코퍼퍼런스; 주의 포화(attention saturation) 노출. |
| NoLiMa | 비어휘적 바늘 | 바늘과 쿼리가 문자적 토큰을 공유하지 않음; 추론 필요.

## 추가 자료

- [Kamradt (2023). Needle in a Haystack 분석](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — 원본 NIAH(Needle in a Haystack) 저장소.
- [Hsieh et al. (2024). RULER: 당신의 장문 컨텍스트 LM의 실제 컨텍스트 크기는?](https://arxiv.org/abs/2404.06654) — 다중 작업 벤치마크.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 실제 장문 컨텍스트 평가.
- [Modarressi et al. (2024). NoLiMa: 비어휘적 니들](https://arxiv.org/abs/2404.06666) — 더 어려운 니들.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — 헤이스택 내 추론.
- [Liu et al. (2024). 중간에 길을 잃다: 언어 모델이 장문 컨텍스트를 사용하는 방식](https://arxiv.org/abs/2307.03172) — 깊이 편향 논문.