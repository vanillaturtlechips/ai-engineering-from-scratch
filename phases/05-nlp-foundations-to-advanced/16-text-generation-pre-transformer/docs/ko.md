# 트랜스포머 이전의 텍스트 생성 — N-그램 언어 모델

> 단어가 놀랍다면 모델은 나쁜 것이다. 퍼플렉서티(perplexity)는 놀라움을 숫자로 만든다. 스무딩(smoothing)은 이를 유한하게 유지한다.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** Phase 5 · 01 (텍스트 처리), Phase 2 · 14 (나이브 베이즈)
**소요 시간:** ~45분

## 문제 정의

트랜스포머(Transformer) 이전, RNN(Recurrent Neural Network) 이전, 단어 임베딩(word embedding) 이전에, 언어 모델은 이전 `n-1` 단어 뒤에 특정 단어가 따라오는 빈도를 세어 다음 단어를 예측했습니다. "the cat" 뒤에 "sat"이 47회, "jumped"가 12회, "refrigerator"가 0회 등장하는 것을 카운트한 후 정규화하여 확률 분포를 얻습니다.

이것이 바로 n-gram 언어 모델입니다. 1980년부터 2015년까지 모든 음성 인식기, 맞춤법 검사기, 구문 기반 기계 번역 시스템을 구동했습니다. 저렴한 온디바이스(on-device) 언어 모델링이 필요할 때 여전히 사용됩니다.

흥미로운 문제는 관찰되지 않은 n-gram을 어떻게 처리할지입니다. 원시 카운트 기반 모델은 본 적 없는 시퀀스에 확률 0을 할당하는데, 이는 치명적입니다. 문장은 길고 거의 모든 긴 문장에는 최소 하나의 미관측 시퀀스가 포함되기 때문입니다. 50년간의 스무딩(smoothing) 연구가 이 문제를 해결했습니다. Kneser-Ney 스무딩이 그 결과이며, 현대 딥러닝은 이 경험적 전통을 계승했습니다.

## 개념

![N-그램 모델: 카운트, 스무딩, 생성](../assets/ngram.svg)

**N-그램 확률:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. `n`을 고정(일반적으로 3은 트라이그램, 4는 4-그램). 카운트로부터 계산:

```text
P(w | context) = count(context, w) / count(context)
```

**제로 카운트 문제.** 훈련에서 보지 못한 모든 n-그램은 확률 0을 얻는다. 2007년 Brown 코퍼스 연구에 따르면 4-그램 모델조차 테스트 4-그램의 30%가 훈련에서 관찰되지 않았다. 스무딩 없이는 실제 텍스트에 대한 평가가 불가능하다.

**정교성 순서별 스무딩 기법:**

1. **라플라스(Add-one).** 모든 카운트에 1을 더함. 간단하지만 희귀 이벤트에 취약.
2. **굿-튜링.** 빈도-빈도를 기반으로 고빈도 이벤트의 확률을 미관측 이벤트로 재할당.
3. **보간법.** n-그램, (n-1)-그램 등의 추정치를 조정 가능한 가중치로 결합.
4. **백오프.** n-그램 카운트가 0이면 (n-1)-그램으로 대체. Katz 백오프는 이를 정규화.
5. **절대 할인.** 모든 카운트에서 고정 할인값 `D`를 빼고 미관측 이벤트에 재분배.
6. **크네저-네이.** 절대 할인 + 하위 차수 모델 선택을 위한 영리한 방법: 원시 빈도 대신 *연속 확률*(단어가 나타나는 컨텍스트 수) 사용.

크네저-네이의 통찰은 심오하다. "San Francisco"는 흔한 바이그램이다. 유니그램 "Francisco"는 대부분 "San" 뒤에 나타난다. 순진한 절대 할인은 "Francisco"에 높은 유니그램 확률을 부여(카운트가 높기 때문). 크네저-네이는 "Francisco"가 단일 컨텍스트에서만 나타남을 인지하고 연속 확률을 낮춘다. 결과: "Francisco"로 끝나는 새로운 바이그램은 적절한 낮은 확률을 얻는다.

**평가: 퍼플렉서티.** 테스트 세트의 단어당 평균 음의 로그 우도의 지수. 낮을수록 좋음. 퍼플렉서티 100은 모델이 100개 단어 중 균일 선택 시만큼 혼란스러움을 의미.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## 구축 방법

### 1단계: 트라이그램 카운트

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

입력은 토큰화된 문장 리스트입니다. 출력은 n-그램 카운트와 컨텍스트 카운트입니다. `<s>`와 `</s>`는 문장 경계 표시입니다.

### 2단계: 라플라스 스무딩

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

모든 카운트에 1을 더합니다. 스무딩을 적용하지만 보이지 않는 이벤트에 과도하게 질량을 할당하여 희귀한 알려진 이벤트에도 영향을 줍니다.

### 3단계: Kneser-Ney (바이그램, 보간)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

세 가지 구성 요소가 있습니다. `continuation_prob`는 "이 단어가 얼마나 다양한 컨텍스트에서 나타나는가?"를 포착합니다(Kneser-Ney의 혁신). `lambda_prev`는 할인에 의해 확보된 질량으로 백오프에 가중치를 부여합니다. 최종 확률은 할인된 주 항과 가중치가 적용된 연속 항의 합입니다.

### 4단계: 샘플링을 통한 텍스트 생성

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

확률에 비례한 샘플링입니다. 시드마다 항상 다른 출력을 제공합니다. 빔 검색과 유사한 출력을 위해 각 단계에서 argmax를 선택하고(탐욕적) 작은 무작위성 조절 장치(온도)를 추가할 수 있습니다.

### 5단계: 퍼플렉서티

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

낮을수록 좋습니다. Brown 코퍼스에서 잘 조정된 4-그램 KN 모델은 퍼플렉서티 약 140을 기록합니다. 트랜스포머 언어 모델은 동일 테스트 세트에서 15-30을 기록합니다. 이 격차는 약 10배이며, 이 격차 때문에 분야가 발전했습니다.

## 사용 사례

- **전통적 NLP 교육.** 스무딩(smoothing), 최대우도추정(MLE), 퍼플렉서티(perplexity)에 대한 가장 명확한 설명.
- **KenLM.** 프로덕션용 N-그램 라이브러리. 낮은 지연 시간이 중요한 음성 및 기계 번역(MT) 시스템에서 재점수화기(rescorer)로 사용됨.
- **온디바이스 자동완성.** 키보드의 트라이그램(trigram) 모델. 여전히 사용 중.
- **베이스라인.** 신경망 언어 모델(neural LM)이 우수하다고 선언하기 전에 항상 N-그램 언어 모델(n-gram LM) 퍼플렉서티(perplexity)를 계산하세요. 트랜스포머(transformer)가 KN(켄링 스무딩, Kneser-Ney smoothing)을 큰 차이로 능가하지 못한다면 문제가 있는 것입니다.

## Ship It

`outputs/prompt-lm-baseline.md`로 저장:

```markdown
---
name: lm-baseline
description: 신경망 언어 모델(LM) 학습 전에 재현 가능한 n-그램 언어 모델 기준선을 구축합니다.
phase: 5
lesson: 16
---

코퍼스와 대상 사용 사례(다음 단어 예측, 재점수화, 퍼플렉서티 기준선)가 주어졌을 때, 다음을 출력합니다:

1. N-그램 차수. 일반 영어에는 트라이그램(trigram), 코퍼스가 큰 경우 4-그램, 음성 재점수화에는 5-그램을 사용합니다.
2. 스무딩. 기본값은 수정된 Kneser-Ney(Modified Kneser-Ney)이며, 교육용으로는 라플라스(Laplace)만 사용합니다.
3. 라이브러리. 프로덕션에는 `kenlm`, 교육용에는 `nltk.lm`을 사용하며, 학습 목적일 때만 직접 구현합니다.
4. 평가. 훈련 및 테스트 세트 간 일관된 토큰화를 적용한 홀드아웃(held-out) 퍼플렉서티(perplexity)를 사용합니다.

비교 대상 시스템 간 다른 토큰화로 계산된 퍼플렉서티 보고는 거부합니다 — 퍼플렉서티 수치는 동일한 토큰화 조건에서만 비교 가능합니다. 테스트 세트의 OOV(Out-Of-Vocabulary) 비율을 표시해야 합니다. Kneser-Ney는 훈련 시 특수 <UNK> 토큰을 예약하지 않는 한 OOV를 잘 처리하지 못합니다.
```

## 연습 문제

1. **쉬움.** 1,000개 문장으로 구성된 셰익스피어 말뭉치에 대해 트라이그램 언어 모델(LM)을 학습시켜라. 20개의 문장을 생성해 보라. 문장들은 지역적으로는 타당하지만 전역적으로는 일관성이 없을 것이다. 이는 표준 데모 예제이다.
2. **중간.** 보유된 셰익스피어 분할 말뭉치에 대해 KN 모델의 퍼플렉서티(perplexity)를 구현하라. 라플라스(Laplace) 방식과 비교하라. KN 모델이 퍼플렉서티를 30-50% 더 낮게 만드는 것을 확인할 수 있을 것이다.
3. **어려움.** 트라이그램 맞춤법 검사기를 구축하라: 오타가 있는 단어와 해당 문맥을 입력으로 받아, 수정 후보들을 생성하고 언어 모델(LM) 하의 문맥 확률로 순위를 매겨라. Birkbeck 맞춤법 말뭉치(공개용)로 평가하라.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| N-그램 | 단어 시퀀스 | `n`개의 연속된 토큰 시퀀스. |
| 스무딩 | 0 방지 | 보이지 않는 사건에 0이 아닌 확률이 할당되도록 확률 질량을 재할당. |
| 퍼플렉서티 | 언어 모델 품질 지표 | `exp(-평균 로그 확률)`. 낮을수록 좋음. |
| 백오프 | 짧은 컨텍스트로 폴백 | 트라이그램 카운트가 0이면 바이그램 사용. Katz 백오프가 이를 공식화. |
| 크네저-네이 | N-그램 최적 스무딩 | 절댓값 할인(absolute discounting) + 하위 모델 연속 확률(continuation probability). |
| 연속 확률 | KN 전용 | `P(w)`를 `w`가 나타나는 컨텍스트 수로 가중치 부여. 원시 카운트(raw count) 기준 아님. |

## 추가 자료

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram 언어 모델(Language Model)과 스무딩(smoothing)에 대한 표준적인 설명.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — Kneser-Ney를 최고의 n-gram 스무딩 기법으로 확립한 논문.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — Kneser-Ney(KN) 기법의 원본 논문.
- [KenLM](https://kheafield.com/code/kenlm/) — 빠른 프로덕션 n-gram 언어 모델(Language Model), 2026년 현재까지도 지연 시간(latency)에 민감한 애플리케이션에서 사용됨.