# 감성 분석

> NLP의 대표적인 작업. 전통적인 텍스트 분류에 대해 알아야 할 대부분의 내용이 여기에 나타납니다.

**유형:** 구축
**언어:** Python
**선수 지식:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (나이브 베이즈)
**소요 시간:** ~75분

## 문제 정의

"음식이 별로였어요." 긍정일까, 부정일까?

감성 분석은 단순해 보입니다. 리뷰어가 무언가를 좋아하거나 좋아하지 않는다고 말했으니 문장에 레이블을 붙이면 됩니다. 이것이 NLP의 대표적인 과제가 된 이유는 모든 쉬워 보이는 사례에 어려운 경우가 숨어 있기 때문입니다. 부정어는 의미를 뒤집습니다. 풍자는 감정을 반전시킵니다. "전혀 나쁘지 않았어요"는 두 개의 부정적 단어가 포함되었지만 긍정입니다. 이모티콘은 주변 텍스트보다 더 강한 신호를 전달합니다. 도메인별 어휘가 중요합니다(음악 리뷰에서의 `tight`와 패션 리뷰에서의 `tight`는 다른 의미).

감성 분석은 전통적인 NLP의 실험장입니다. 모든 단순한 베이스라인이 특정 실패 모드를 보이는 이유를 이해한다면, 더 복잡한 모델이 왜 발명되었는지 알 수 있습니다. 이 강의에서는 Naive Bayes 베이스라인을 처음부터 구축하고, 로지스틱 회귀를 추가한 뒤, 프로덕션 수준의 감성 분석을 규정 준수 등급으로 만드는 함정들을 설명합니다.

## 개념

![감정 분석 파이프라인: 토큰 → 특징 → 분류기 → 레이블](./assets/sentiment.svg)

전통적인 감정 분석은 두 단계로 이루어집니다.

1. **표현.** 텍스트를 특징 벡터로 변환합니다. BoW(Bag of Words), TF-IDF, 또는 n-그램(n-gram)을 사용합니다.
2. **분류.** 레이블이 지정된 예시에 선형 모델(나이브 베이즈, 로지스틱 회귀, SVM)을 적용합니다.

나이브 베이즈는 작동하는 가장 단순한 모델입니다. 레이블이 주어졌을 때 모든 특징이 독립적이라고 가정합니다. `P(단어 | 긍정)`과 `P(단어 | 부정)`을 카운트로부터 추정합니다. 추론 시 확률을 곱합니다. "나이브"한 독립성 가정은 터무니없이 잘못되었지만 결과는 놀랍도록 강력합니다. 그 이유: 희소한 텍스트 특징과 중간 규모의 데이터에서는 분류기가 각 단어가 어느 쪽으로 더 기울어지는지에 더 관심을 가지기 때문입니다.

로지스틱 회귀는 독립성 가정을 수정합니다. 특징마다 가중치를 학습하며, 음의 가중치도 포함합니다. "not good" 같은 바이그램 특징은 음의 가중치를 받습니다. 나이브 베이즈는 레이블을 지정하지 않은 바이그램에 대해서는 이를 수행할 수 없습니다.

## 구축 방법

### 1단계: 실제 미니 데이터셋

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

의도적으로 작게 구성했습니다. 실제 작업에서는 수만 개의 예시(IMDb, SST-2, Yelp 극성)를 사용합니다. 수학적 원리는 동일합니다.

### 2단계: 처음부터 시작하는 다항 나이브 베이즈

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

가산성 평활화(alpha=1.0)는 라플라스 평활화입니다. 이 평활화가 없으면 어떤 클래스에서 한 번도 등장하지 않은 단어의 확률이 0이 되어 로그 값이 발산합니다. 실제 작업에서는 `alpha=0.01`이 일반적입니다. `alpha=1.0`은 교육용 기본값입니다.

### 3단계: 처음부터 시작하는 로지스틱 회귀

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 정규화는 여기서 중요합니다. 텍스트 특징은 희소(sparse)하기 때문에 L2 정규화가 없으면 모델이 훈련 예시를 암기하게 됩니다. `0.01`로 시작하여 튜닝하세요.

### 4단계: 부정 처리(실패 모드)

"not good"과 "not bad"를 고려해봅시다. BoW(Bag of Words) 분류기는 `{not, good}`과 `{not, bad}`를 보고 훈련 데이터에서 더 많이 나타난 것을 학습합니다. 바이그램 분류기는 `not_good`과 `not_bad`를 별개의 특징으로 보고 학습합니다. 일반적으로 이 정도로 충분합니다.

바이그램을 사용하지 않을 때 작동하는 더 단순한 해결책: **부정 범위 지정**. 부정 단어 뒤에 오는 토큰에 `NOT_` 접두사를 붙이고 다음 구두점까지를 범위로 합니다.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

이제 `good`과 `NOT_good`은 서로 다른 특징입니다. 분류기가 이 둘을 반대 가중치로 학습할 수 있습니다. 3줄의 전처리 코드로 감성 벤치마크에서 측정 가능한 정확도 향상을 얻을 수 있습니다.

### 5단계: 중요한 평가 지표

클래스가 불균형할 때 정확도만으로는 오해의 소지가 있습니다. 실제 감성 코퍼스는 보통 70-80%가 긍정 또는 70-80%가 부정입니다. 다수 클래스만 예측하는 분류기가 80% 정확도를 얻어도 쓸모없습니다. 다음 지표들을 모두 보고하세요:

- **클래스별 정밀도와 재현율.** 클래스당 한 쌍씩. 클래스 균형을 고려한 단일 숫자를 얻으려면 매크로 평균을 구하세요.
- **매크로-F1(불균형 데이터용 주요 지표).** 클래스별 F1 점수의 평균으로, 동일한 가중치를 적용합니다. 클래스가 불균형할 때 정확도 대신 사용하세요.
- **가중치-F1(대체 지표).** 매크로와 동일하지만 클래스 빈도에 따라 가중치를 적용합니다. 불균형 자체가 비즈니스적 의미를 가질 때 매크로-F1과 함께 보고하세요.
- **혼동 행렬.** 원시 개수. 스칼라 지표를 신뢰하기 전에 항상 검사하세요. 모델이 어떤 클래스 쌍을 혼동하는지 알 수 있습니다.
- **클래스별 오류 샘플.** 클래스당 5개의 잘못된 예측을 추출하세요. 직접 읽어보세요. 실제 오류를 읽는 것만큼 좋은 방법은 없습니다.

심하게 불균형한 데이터(> 95-5 비율)의 경우 정확도 대신 **AUROC**와 **AUPRC**를 보고하세요. AUPRC는 일반적으로 관심 있는 소수 클래스(스팸, 사기, 드문 감성)에 더 민감합니다.

**피해야 할 일반적인 오류.** 불균형 데이터에서 마이크로-F1을 매크로-F1 대신 보고하면 다수 클래스에 의해 지배되어 높은 숫자가 나옵니다. 매크로-F1은 소수 클래스 성능을 반드시 확인하도록 강제합니다.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 사용 방법

scikit-learn은 단 6줄로 정확하게 구현합니다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

주목할 세 가지 사항. `stop_words=None`은 부정어를 보존합니다. `ngram_range=(1, 2)`는 바이그램을 추가하여 `not_good`을 특징으로 만듭니다. `sublinear_tf=True`는 반복된 단어를 약화시킵니다. 이 세 가지 플래그는 SST-2에서 75% 정확도의 베이스라인과 85% 정확도의 베이스라인 차이를 만듭니다.

### 트랜스포머를 선택해야 할 때

- **풍자 감지**. 클래식 모델은 여기서 실패합니다. 마침표.
- **문서 중간에 감정이 변하는 긴 리뷰**.
- **측면 기반 감정 분석**. "카메라는 훌륭했지만 배터리는 끔찍했어요." 감정을 측면에 할당해야 합니다. 트랜스포머 또는 구조화된 출력 모델만 가능.
- **비영어권, 저자원 언어**. 다국어 BERT는 무료로 제로샷 베이스라인을 제공합니다.

위 사항 중 하나라도 필요하면 7단계(트랜스포머 심층 분석)로 건너뛰세요. 그렇지 않으면 TF-IDF + 바이그램 + 부정어 처리를 적용한 나이브 베이즈 또는 로지스틱 회귀가 2026년 프로덕션 베이스라인입니다.

### 재현성 함정 (다시)

감정 모델 재학습은 일상적입니다. 재평가는 그렇지 않습니다. 논문에서 보고된 정확도 수치는 특정 분할, 특정 전처리, 특정 토크나이저를 사용합니다. 동일한 파이프라인을 사용하지 않고 새 모델을 베이스라인과 비교하면 오해의 소지가 있는 결과가 나옵니다. 항상 논문의 수치가 아닌 자신의 파이프라인에서 베이스라인을 재생성하세요.

## Ship It

`outputs/prompt-sentiment-baseline.md`로 저장:

```markdown
---
name: sentiment-baseline
description: 새로운 데이터셋에 대한 감성 분석 기준 모델을 설계합니다.
phase: 5
lesson: 05
---

데이터셋 설명(도메인, 언어, 크기, 레이블 세분화 정도, 지연 시간 예산)이 주어졌을 때, 다음을 출력합니다:

1. **특징 추출 레시피**. 토크나이저, n-gram 범위, 불용어 정책(보통 유지), 부정 처리 방식(범위 접두사 또는 바이그램)을 명시합니다.
2. **분류기**. 기준 모델에는 나이브 베이즈(Naive Bayes), 프로덕션에는 로지스틱 회귀(Logistic Regression), 도메인에서 반어/측면/다국어 분석이 필요한 경우에만 트랜스포머(Transformer)를 사용합니다.
3. **평가 계획**. 정밀도(Precision), 재현율(Recall), F1 점수, 혼동 행렬(Confusion Matrix), 클래스별 오류 샘플(스칼라 값만 보고하지 않음)을 보고합니다.
4. **배포 후 모니터링할 실패 모드**. 도메인 드리프트(Domain Drift)와 반어(Sarcasm)가 상위 2개입니다.

감성 작업에서 불용어 제거를 권장하지 않습니다. 클래스가 불균형할 때(예: 90% 긍정) 정확도(Accuracy)를 유일한 지표로 보고하지 않습니다. 부분어(subword)가 풍부한 언어는 단어 수준 TF-IDF 대신 FastText 또는 트랜스포머 임베딩(Transformer Embedding)이 필요함을 표시합니다.
```

## 연습 문제

1. **쉬움.** `apply_negation`을 scikit-learn 파이프라인의 전처리 단계로 추가하고, 작은 감정 분석 데이터셋에서 F1 점수 변화(delta)를 측정하세요.  
2. **중간.** 클래스 가중치 적용 로지스틱 회귀를 구현하세요 (scikit-learn에 `class_weight="balanced"` 전달, 또는 직접 그래디언트 유도). 합성 90-10 클래스 불균형 데이터에서 효과를 측정하세요.  
3. **어려움.** 감정 모델의 잔차(residuals)에 두 번째 분류기를 훈련시켜 sarcasm 감지기(sarcasm detector)를 구축하세요. 실험 설정을 문서화하세요. 정확도가 무작위 수준(chance-level) 미만일 때 독자에게 경고하세요 (2-class sarcasm에서 무작위 수준은 ~50%이며, 대부분의 첫 시도는 이 수준에 머뭅니다).

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 극성(Polarity) | 긍정 또는 부정 | 이진 레이블; 때로는 중립 또는 세분화된(5점) 평가로 확장됨. |
| 측면 기반 감성(Aspect-based sentiment) | 측면별 극성 | 텍스트에서 언급된 특정 개체 또는 속성에 감성을 할당함. |
| 부정 범위 지정(Negation scoping) | 근처 토큰 반전 | "not" 이후의 토큰에 `NOT_` 접두사를 구두점까지 추가함. |
| 라플라스 평활화(Laplace smoothing) | 카운트에 1 추가 | 나이브 베이즈에서 0-확률 특성 방지. |
| L2 정규화(L2 regularization) | 가중치 축소 | 손실에 `lambda * sum(w^2)` 추가. 희소 텍스트 특성에 필수적. |

## 추가 자료

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 기초적인 조사 논문. 분량이 길지만 처음 4개 섹션에서 전통적인 내용을 모두 다룹니다.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 짧은 텍스트에서 빅램(bigram) + 나이브 베이즈(Naive Bayes)를 능가하기 어렵다는 것을 보여준 논문.
- [scikit-learn 텍스트 특성 추출 문서](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`, `TfidfVectorizer` 및 조정할 수 있는 모든 파라미터에 대한 참고 자료.