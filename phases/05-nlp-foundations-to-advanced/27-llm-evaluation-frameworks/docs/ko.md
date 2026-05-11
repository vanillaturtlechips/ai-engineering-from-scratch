# LLM 평가 — RAGAS, DeepEval, G-Eval

> Exact-match와 F1은 의미적 동등성을 놓칩니다. 인간 검토는 확장성이 없습니다. LLM-as-judge가 프로덕션 답변입니다 — 숫자를 신뢰할 수 있을 만큼 충분한 보정이 필요합니다.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** Phase 5 · 13 (질문 답변), Phase 5 · 14 (정보 검색)
**소요 시간:** ~75분

## 문제

당신의 RAG 시스템은 다음과 같이 답변합니다: "June 29th, 2007."  
금참(gold reference)은 다음과 같습니다: "June 29, 2007."  
정확 일치(Exact Match) 점수는 0점입니다. F1 점수는 약 75%입니다. 인간은 100% 점수를 줄 것입니다.

이제 10,000개의 테스트 케이스로 확장합니다. 검색기(retriever), 청킹(chunking), 프롬프트(prompt), 모델(model)의 모든 변경 사항마다 이 과정을 반복합니다. 의미를 이해하고, 대규모로 저렴하게 실행되며, 회귀(regression)에 대해 거짓을 말하지 않으며, 올바른 실패 모드를 표면화하는 평가기가 필요합니다.

2026년에는 이 문제를 해결하는 세 가지 프레임워크가 있습니다.

- **RAGAS.** 검색 증강 생성 평가(Retrieval-Augmented Generation ASsessment). NLI(자연어 추론) + LLM-심사관(LLM-judge) 백엔드를 사용하는 4가지 RAG 메트릭(충실도(faithfulness), 답변 관련성(answer-relevance), 컨텍스트 정밀도(context-precision), 컨텍스트 재현율(context-recall)). 연구 기반, 경량화.
- **DeepEval.** LLM을 위한 Pytest. G-Eval, 작업 완료(task-completion), 환각(hallucination), 편향(bias) 메트릭. CI/CD 네이티브.
- **G-Eval.** 방법론과 (DeepEval 메트릭): 연쇄 사고(chain-of-thought), 사용자 정의 기준, 0-1 점수를 사용하는 LLM-심사관(LLM-as-judge).

세 가지 모두 LLM-심사관(LLM-as-judge)에 의존합니다. 이 레슨은 이 방법론과 그 주변의 신뢰 계층(trust layer)에 대한 직관을 구축합니다.

## 개념

![네 가지 평가 차원, LLM-as-judge 아키텍처](../assets/llm-evaluation.svg)

**LLM-as-judge.** 정적 메트릭을 평가 기준(rubric)을 바탕으로 출력을 점수화하는 LLM으로 대체합니다. `(쿼리, 컨텍스트, 답변)`이 주어지면, 평가자 LLM에 다음과 같이 프롬프트를 제공합니다: "충실도(faithfulness)를 0-1점으로 평가하세요." 점수를 반환합니다.

작동 원리: LLM은 인간 판단을 극히 낮은 비용으로 근사화합니다. GPT-4o-mini는 평가 사례당 약 $0.003으로, 1000개 샘플 회귀 평가 실행을 $5 미만으로 가능하게 합니다.

실패하는 이유 (무증상):

1. **평가자 편향.** 평가자는 더 긴 답변, 자신의 모델 패밀리가 생성한 답변, 프롬프트 스타일과 일치하는 답변을 선호합니다.
2. **JSON 파싱 실패.** 잘못된 JSON → NaN 점수 → 집계 시 무증상으로 제외됩니다. RAGAS 사용자는 이 문제를 잘 알고 있습니다. try/except + 명시적 실패 모드로 게이트를 설정하세요.
3. **모델 버전 간 드리프트.** 평가자 업그레이드 시 모든 메트릭이 변경됩니다. 평가자 모델과 버전을 고정하세요.

**RAG의 네 가지 메트릭.**

| 메트릭 | 질문 | 백엔드 |
|--------|----------|---------|
| 충실도(Faithfulness) | 답변의 각 주장이 검색된 컨텍스트에서 비롯되었는가? | NLI 기반 함의 분석 |
| 답변 관련성(Answer relevance) | 답변이 질문을 다루었는가? | 답변에서 가상의 질문 생성; 실제 질문과 비교 |
| 컨텍스트 정밀도(Context precision) | 검색된 청크 중 관련 있는 비율은 얼마인가? | LLM 평가자 |
| 컨텍스트 재현율(Context recall) | 검색이 필요한 모든 것을 반환했는가? | 정답 답변에 대한 LLM 평가자 |

**G-Eval.** 사용자 정의 기준 정의: "답변이 올바른 출처를 인용했는가?" 프레임워크는 자동으로 사고 연쇄 평가 단계로 확장한 후 0-1점을 부여합니다. RAGAS가 커버하지 않는 도메인별 품질 차원에 적합합니다.

**보정(Calibration).** 인간 라벨과의 상관관계가 확인되기 전까지 원시 평가자 점수를 신뢰하지 마세요. 100개의 수작업 라벨 예제를 실행합니다. 평가자 vs 인간 점수를 플롯하고 스피어만 상관계수(rho)를 계산합니다. rho < 0.7인 경우 평가자 기준을 개선해야 합니다.

## 구축 방법

### 1단계: NLI를 통한 충실도 평가 (RAGAS 스타일)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm`은 모든 호출 가능한 함수: 프롬프트 str -> 생성된 str.
# 예: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""이 답변을 단순한 사실 주장으로 분리하세요(줄당 하나씩):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

답변을 원자적 주장으로 분해합니다. NLI를 사용하여 각 주장을 검색된 컨텍스트와 비교합니다. 충실도 = 지원되는 주장의 비율.

### 2단계: 답변 관련성 평가

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: .encode(texts, normalize_embeddings=True) -> ndarray를 구현하는 모든 모델
# 예: encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"이 답변이 답이 될 수 있는 {n}개의 질문을 작성하세요:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

답변이 질문한 내용과 다른 질문을 암시하면 관련성이 감소합니다.

### 3단계: G-Eval 사용자 정의 메트릭

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="정확성",
    criteria="답변은 사실적으로 정확해야 하며 예상 출력과 일치해야 합니다.",
    evaluation_steps=[
        "예상 출력을 읽습니다.",
        "실제 출력을 읽습니다.",
        "실제 출력의 사실적 주장을 나열합니다.",
        "각 주장에 대해 예상 출력에 의해 지원되는지 여부를 표시합니다.",
        "점수 = 지원되는 비율 반환.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="첫 번째 아이폰은 언제 출시되었나요?",
                   actual_output="2007년 6월 29일.",
                   expected_output="2007년 6월 29일.")
metric.measure(test)
print(metric.score, metric.reason)
```

평가 단계는 평가 기준입니다. 명시적인 단계는 암시적 "0-1 점수" 프롬프트보다 더 안정적입니다.

### 4단계: CI 게이트

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"{case.id}에서 충실도 회귀 발생"
        rel.measure(case)
        assert rel.score >= 0.7, f"{case.id}에서 관련성 회귀 발생"
```

pytest 파일로 배포합니다. 모든 PR에서 실행합니다. 회귀 발생 시 병합을 차단합니다.

### 5단계: 처음부터 시작하는 간단한 평가

`code/main.py` 참조. 충실도(답변 주장과 컨텍스트의 중복) 및 관련성(답변 토큰과 질문 토큰의 중복)의 표준 라이브러리 전용 근사치. 프로덕션용은 아닙니다. 구조를 보여줍니다.

## 함정

- **보정(calibration) 없음.** 인간 라벨과 0.3 상관관계를 보이는 심사자는 노이즈입니다. 출시 전 보정 실행을 필수로 하세요.
- **자기 평가(self-evaluation).** 동일한 LLM을 생성 및 평가에 사용하면 점수가 10-20% 과장됩니다. 심사에는 다른 모델 패밀리를 사용하세요.
- **쌍별 평가의 위치 편향(positional bias).** 심사자는 먼저 제시된 옵션을 선호합니다. 항상 순서를 무작위화하고 양방향으로 실행하세요.
- **원시 집계는 실패 사례를 가립니다.** 평균 점수 0.85는 종종 5%의 치명적 실패 사례를 숨깁니다. 항상 하위 분위(bottom quantile)를 검사하세요.
- **골든 데이터셋 열화(golden dataset rot).** 시간이 지남에 따라 변하는 버전 관리되지 않은 평가 세트는 종단적 비교를 방해합니다. 모든 변경 사항에 데이터셋 태그를 추가하세요.
- **LLM 비용.** 대규모 평가 시 심사 호출이 비용을 지배합니다. 보정 임계값을 충족하는 가장 저렴한 모델을 사용하세요. GPT-4o-mini, Claude Haiku, Mistral-small 등.

## 사용 방법

2026 스택:

| 사용 사례 | 프레임워크 |
|---------|-----------|
| RAG 품질 모니터링 | RAGAS (4개 메트릭) |
| CI/CD 회귀 게이트 | DeepEval + pytest |
| 커스텀 도메인 기준 | DeepEval 내 G-Eval |
| 온라인 실시간 트래픽 모니터링 | 참조 문서 없는 모드 RAGAS |
| 인간 개입 스팟 체크 | 주석 UI가 포함된 LangSmith 또는 Phoenix |
| 레드 팀 / 안전성 평가 | Promptfoo + DeepEval |

일반적인 스택: 모니터링에는 RAGAS, CI에는 DeepEval, 새로운 차원 평가에는 G-Eval 사용. 세 가지 모두 실행; 유용하게 결과가 상이함.

## Ship It

`outputs/skill-eval-architect.md`로 저장:

```markdown
---
name: eval-architect
description: 보정된 평가자와 CI 게이트를 활용한 LLM 평가 계획 설계.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

사용 사례(RAG / 에이전트 / 생성 작업)가 주어졌을 때 다음을 출력:

1. 메트릭. 충실도(faithfulness) / 관련성(relevance) / 컨텍스트 정밀도(context-precision) / 컨텍스트 재현율(context-recall) + 기준(criteria)이 포함된 사용자 정의 G-Eval 메트릭.
2. 평가자 모델. 모델명 + 버전, 비용 대비 정확도 선택 근거.
3. 보정. 수작업 라벨링 세트 크기, 인간 평가 대비 목표 스피어만 상관계수(Spearman rho) > 0.7.
4. 데이터셋 버전 관리. 태깅 전략, 변경 로그, 층화(stratification) 전략.
5. CI 게이트. 메트릭별 임계값, 회귀 감지 기간(regression-window) 로직, 하위 백분위수(bottom-quantile) 경고.

50개 이상의 인간 라벨링 예시와 검증되지 않은 평가자 사용을 거부. 자기 평가(동일 모델이 생성 및 평가) 거부. 하위 10% 사례 분석 없이 집계만 보고하는 것을 거부. 평가자 모델 업그레이드가 병렬 기준선 평가 없이 적용되는 파이프라인을 경고.
```

## 연습 문제

1. **쉬움.** 알려진 환각(hallucination)이 포함된 10개의 RAG 예제에 RAGAS를 적용하세요. 충실도(faithfulness) 메트릭이 각각의 환각을 포착하는지 확인하세요.
2. **중간.** 50개의 QA 답변에 대해 0-1 척도로 정답 여부를 수동으로 라벨링하세요. G-Eval로 점수를 매기고, 평가자 점수와 인간 점수 간의 스피어만 상관 계수(Spearman rho)를 측정하세요.
3. **어려움.** DeepEval을 사용해 pytest CI 게이트를 구축하세요. 의도적으로 검색기(retriever)를 성능 저하시킨 후 게이트가 실패하는지 확인하세요. 하위 10% 분위수(quantile)에 대한 임계값(threshold) 검사를 통해 경고 시스템을 추가하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| LLM-as-judge | LLM으로 평가 | 평가 기준(rubric)을 바탕으로 판사 모델(judge model)에 프롬프트를 입력하여 출력값을 0-1 점수로 평가 |
| RAGAS | RAG 메트릭 라이브러리 | 4가지 참조(reference)가 필요 없는 RAG 메트릭을 포함한 오픈소스 평가 프레임워크 |
| Faithfulness | 답변이 근거 기반인가? | 검색된 컨텍스트에 의해 뒷받침되는 답변 주장의 비율 |
| Context precision | 검색된 청크가 관련성이 있었는가? | 실제로 중요했던 상위-K 청크의 비율 |
| Context recall | 검색이 모든 것을 찾았는가? | 검색된 청크에 의해 지원되는 정답 주장의 비율 |
| G-Eval | 커스텀 LLM 평가자 | 평가 기준 + 사고 연쇄(chain-of-thought) 평가 단계 + 0-1 점수 |
| Calibration | 신뢰하되 검증하라 | 평가자 점수와 인간 점수 간의 스피어만 상관계수(Spearman correlation) |

## 추가 자료

- [Es et al. (2023). RAGAS: 검색 증강 생성의 자동화된 평가](https://arxiv.org/abs/2309.15217) — RAGAS 논문.
- [Liu et al. (2023). G-Eval: GPT-4를 활용한 자연어 생성 평가 및 인간 평가와의 정렬 개선](https://arxiv.org/abs/2303.16634) — G-Eval 논문.
- [DeepEval 문서](https://deepeval.com/docs/metrics-introduction) — 오픈소스 프로덕션 스택.
- [Zheng et al. (2023). MT-Bench와 Chatbot Arena를 활용한 LLM-as-a-Judge 평가](https://arxiv.org/abs/2306.05685) — 편향, 보정, 한계.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — RAGAS, DeepEval, Phoenix를 통합하는 프레임워크.