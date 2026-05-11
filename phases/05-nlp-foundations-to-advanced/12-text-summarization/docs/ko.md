# 텍스트 요약

> 추출 기반 시스템은 문서가 말한 내용을 알려줍니다. 생성 기반 시스템은 저자가 의도한 바를 알려줍니다. 다른 작업, 다른 함정.

**유형:** 구축
**언어:** Python
**선수 지식:** 5단계 · 02 (BoW + TF-IDF), 5단계 · 11 (기계 번역)
**소요 시간:** ~75분

## 문제 정의

2,000단어 분량의 뉴스 기사가 피드에 도착했다. 이를 요약하는 120단어가 필요하다. 기사에서 가장 중요한 세 문장을 선택하거나(추출적), 내용을 자신의 말로 재구성할 수 있다(생성적). 둘 다 요약이라고 부르지만 완전히 다른 문제다.

추출적 요약은 순위 결정 문제다. 모든 문장에 점수를 매기고 상위 `k`개를 반환한다. 출력은 항상 문법적으로 올바르지만, 기사 전체에 분산된 내용을 놓칠 위험이 있다.

생성적 요약은 생성 문제다. 트랜스포머가 입력에 조건화된 새로운 텍스트를 생성한다. 출력은 유창하고 압축적이지만, 출처에 없는 사실을 환각(hallucination)할 수 있다. 위험은 확신에 찬 허위 생성이다.

이 강의에서는 각각의 실패 모드를 포함해 두 방법을 모두 다룬다.

## 개념

![추출적 TextRank vs 추상적 트랜스포머](../assets/summarization.svg)

**추출적(Extractive).** 기사를 문장 노드(node)와 유사도 간선(edge)로 구성된 그래프로 처리합니다. 그래프 위에서 PageRank(또는 유사 알고리즘)를 실행해 문장들이 전체와 얼마나 연결되었는지에 따라 점수를 매깁니다. 가장 높은 점수를 받은 문장들이 요약문이 됩니다. 대표적인 구현체는 **TextRank**(Mihalcea and Tarau, 2004)입니다.

**추상적(Abstractive).** 문서-요약 쌍으로 트랜스포머 인코더-디코더(BART, T5, Pegasus)를 파인튜닝(fine-tuning)합니다. 추론(inference) 시 모델은 문서를 읽고 크로스-어텐션(cross-attention)을 통해 토큰 단위로 요약문을 생성합니다. 특히 Pegasus는 갭-문장(gap-sentence) 사전학습 목표를 사용해 적은 파인튜닝으로도 요약에 탁월합니다.

**ROUGE**(Recall-Oriented Understudy for Gisting Evaluation)로 평가합니다. ROUGE-1과 ROUGE-2는 유니그램 및 바이그램 중첩을 측정합니다. ROUGE-L은 가장 긴 공통 부분 수열(longest common subsequence)을 평가합니다. 점수가 높을수록 좋으며, ROUGE-L 40은 "좋음", 50은 "탁월함"으로 간주됩니다. 모든 논문에서 세 가지 지표를 보고합니다. `rouge-score` 패키지를 사용하세요.

## 구축 방법

### 1단계: TextRank (추출적 요약)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

두 가지 주목할 점. 유사도 함수는 로그 정규화된 단어 중복을 사용하며, 이는 원래 TextRank 변형입니다. TF-IDF 벡터의 코사인 유사도도 작동합니다. 감쇠 계수 0.85와 반복 횟수는 PageRank 기본값입니다.

### 2단계: BART를 이용한 생성적 요약

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(긴 뉴스 기사 텍스트)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN은 CNN/DailyMail 코퍼스에 대해 파인튜닝(fine-tuning)되었습니다. 뉴스 스타일의 요약을 바로 생성합니다. 다른 도메인(과학 논문, 대화, 법률)의 경우 해당 Pegasus 체크포인트를 사용하거나 대상 데이터에 대해 파인튜닝(fine-tuning)하세요.

### 3단계: ROUGE 평가

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

항상 어간 추출(stemming)을 사용하세요. 어간 추출이 없으면 "running"과 "run"이 다른 단어로 처리되어 ROUGE가 과소평가됩니다.

### ROUGE 그 이후 (2026년 요약 평가)

ROUGE는 20년 동안 지배적인 요약 평가 지표였지만 2026년 현재만으로는 부족합니다. 대규모 NLG 논문 메타 분석에 따르면:

- **BERTScore** (문맥 임베딩 유사도)는 2023년까지 성장했으며, 현재 대부분의 요약 논문에서 ROUGE와 함께 보고됩니다.
- **BARTScore**는 평가를 생성으로 처리합니다: 사전 훈련된 BART가 소스에 대해 할당한 확률을 기반으로 요약 점수를 매깁니다.
- **MoverScore** (문맥 임베딩에 대한 Earth Mover's Distance)는 2025년 요약 벤치마크에서 ROUGE보다 의미적 중복을 더 잘 포착하여 1위를 차지했습니다.
- **FactCC**와 **QA 기반 충실도**는 2021-2023년에 흔했으나, 현재는 **G-Eval** (GPT-4 프롬프트 체인으로 일관성, 일관성, 유창성, 관련성을 체이브 오브 사고 추론으로 평가)로 대체되는 경우가 많습니다.
- **G-Eval** 및 유사한 LLM-평가기 접근법은 평가 기준이 잘 설계되었을 때 인간 판단과 약 80% 일치합니다.

프로덕션 권장 사항: 레거시 비교를 위해 ROUGE-L, 의미적 중복을 위해 BERTScore, 일관성 및 사실성을 위해 G-Eval을 보고하세요. 50-100개의 인간 라벨 요약에 대해 보정하세요.

### 4단계: 사실성 문제

생성적 요약은 환각(hallucination)에 취약합니다. 추출적 요약은 출력이 소스에서 그대로 가져오기 때문에 환각 위험이 훨씬 낮지만, 소스 문장이 문맥에서 벗어나거나 오래되었거나 순서가 바뀌어 인용되면 여전히 오해를 일으킬 수 있습니다. 이는 규정 준수 관련 콘텐츠에 대해 프로덕션 시스템이 여전히 추출적 방법을 선호하는 가장 큰 이유입니다.

명명할 환각 유형:

- **엔티티 교체.** 소스는 "John Smith"라고 말합니다. 요약은 "John Brown"이라고 말합니다.
- **숫자 변동.** 소스는 "25,000"이라고 말합니다. 요약은 "25 million"이라고 말합니다.
- **극성 반전.** 소스는 "제안을 거부했다"고 말합니다. 요약은 "제안을 수락했다"고 말합니다.
- **사실 발명.** 소스는 CEO를 언급하지 않습니다. 요약은 CEO가 승인했다고 말합니다.

효과적인 평가 접근법:

- **FactCC.** 소스 문장과 요약 문장 간의 함의에 대해 훈련된 이진 분류기. 사실/비사실 예측.
- **QA 기반 사실성.** QA 모델에 소스에 답이 있는 질문을 합니다. 요약이 다른 답을 지원하면 플래그 지정.
- **엔티티 수준 F1.** 소스와 요약의 명명된 엔티티 비교. 요약에만 있는 엔티티는 의심스러움.

사실성이 중요한 사용자 대상 콘텐츠(뉴스, 의료, 법률, 금융)의 경우 추출적 방법이 더 안전한 기본값입니다. 생성적 방법은 루프 내 사실성 검사가 필요합니다.

## 사용 방법

2026년 스택:

| 사용 사례 | 권장 모델 |
|---------|----------|
| 뉴스, 3-5문장 요약, 영어 | `facebook/bart-large-cnn` |
| 과학 논문 | `google/pegasus-pubmed` 또는 튜닝된 T5 |
| 다중 문서, 장문 요약 | 32k+ 컨텍스트를 가진 모든 LLM, 프롬프트 기반 |
| 대화 요약 | `philschmid/bart-large-cnn-samsum` |
| 추출적 요약, 구조적 환각 위험 최소화 | TextRank 또는 `sumy`의 LSA / LexRank |

긴 컨텍스트를 가진 LLM은 계산 제약이 없을 때 2026년 기준 전문 모델들을 종종 능가합니다. 다만 트레이드오프는 비용과 재현성입니다. 전문 모델들은 더 일관된 출력을 제공합니다.

## Ship It

`outputs/skill-summary-picker.md`로 저장:

```markdown
---
name: summary-picker
description: 추출적 또는 생성적 요약, 명명된 라이브러리, 사실성 검증 선택.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, 요약]
---

주어진 작업(문서 유형, 규정 요구사항, 길이, 계산 예산)에 따라 다음을 출력:

1. 접근 방식. 추출적 또는 생성적. 이유를 한 문장으로 설명.
2. 시작 모델/라이브러리. 이름을 명시. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed` 또는 LLM 프롬프트.
3. 평가 계획. ROUGE-1, ROUGE-2, ROUGE-L (어간 추출과 함께 rouge-score 사용). 생성적 요약일 경우 사실성 검증 추가.
4. 탐지할 실패 모드 하나. 생성적 뉴스 요약에서 가장 흔한 것은 엔티티 교체; 소스에 있는 엔티티가 요약에 나타나지 않는 샘플을 표시.

의료, 법률, 금융 또는 규제 대상 콘텐츠에 대해 사실성 검증 없이 생성적 요약을 거부. 모델 컨텍스트 윈도우를 초과하는 입력은 청킹 맵-리듀스 요약이 필요하다고 표시(단순 자르기가 아님).
```

## 연습 문제

1. **쉬움.** 5개의 뉴스 기사에 대해 TextRank를 실행하세요. 상위 3개 문장을 참조 요약과 비교하세요. ROUGE-L을 측정하세요. CNN/DailyMail 스타일의 기사에서 30-45 ROUGE-L 점수를 확인할 수 있을 것입니다.
2. **중간.** 엔티티 수준 사실성 구현: 소스 텍스트와 요약에서 명명된 엔티티를 추출하세요(spaCy), 요약에서 소스 엔티티의 재현율(recall)과 소스에 대한 요약 엔티티의 정밀도(precision)를 계산하세요. 높은 정밀도와 낮은 재현율은 안전하지만 간결한 요약을 의미합니다. 낮은 정밀도는 환각된 엔티티를 의미합니다.
3. **어려움.** 50개의 CNN/DailyMail 기사에서 BART-large-CNN과 LLM(Claude 또는 GPT-4)을 비교하세요. ROUGE-L, 엔티티 F1을 통한 사실성, 요약당 비용을 보고하세요. 각 모델이 우수한 부분을 문서화하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 추출적(Extractive) | 문장 선택 | 소스에서 문장을 그대로 반환. 절대 환각(hallucination)을 일으키지 않음. |
| 생성적(Abstractive) | 다시 쓰기 | 소스를 조건으로 새로운 텍스트 생성. 환각을 일으킬 수 있음. |
| ROUGE | 요약 평가 지표 | 시스템 출력과 참조 요약 간의 N-gram / 최장 공통 부분열(LCS) 일치도. |
| TextRank | 그래프 기반 추출 | 문장 유사도 그래프에 대한 PageRank 적용. |
| 사실성(Factuality) | 맞는지 | 요약 내용이 소스에 의해 뒷받침되는지 여부. |
| 환각(Hallucination) | 만들어진 내용 | 소스가 지원하지 않는 요약 내 내용.

## 추가 자료

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — 추출적 요약의 기본 논문.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 논문.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — PEGASUS 및 갭 문장 목표 논문.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 평가 지표 논문.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 추상적 요약의 사실성 논문.