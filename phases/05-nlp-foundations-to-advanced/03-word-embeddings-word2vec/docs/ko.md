# 워드 임베딩 — 처음부터 구현하는 Word2Vec

> 단어는 함께 어울리는 동료를 가진다. 그 아이디어로 얕은 네트워크를 훈련시키면 기하학이 자연스럽게 드러난다.

**유형:** 구축
**언어:** Python
**선수 지식:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (처음부터 구현하는 역전파)
**소요 시간:** ~75분

## 문제 정의

TF-IDF는 `dog`와 `puppy`가 서로 다른 단어임을 알지만, 이 둘이 거의 같은 의미를 가진다는 것을 알지 못합니다. `dog`로 훈련된 분류기는 `puppy`에 대한 리뷰로 일반화할 수 없습니다. 동의어를 나열하여 이 문제를 일시적으로 해결할 수 있지만, 이는 희귀 용어, 도메인 전문 용어, 그리고 예상하지 못한 모든 언어에서 실패합니다.

`dog`와 `puppy`가 공간에서 서로 가까이 위치하도록 하는 표현 방식을 원합니다. `king - man + woman`이 `queen` 근처에 위치하도록 하는 표현 방식을 원합니다. `dog`로 훈련된 모델이 `puppy`에 대한 신호를 무료로 전이할 수 있는 표현 방식을 원합니다.

Word2Vec은 이러한 공간을 제공했습니다. 2계층 신경망, 1조 토큰 훈련 실행, 2013년 발표. 이 아키텍처는 거의 창피할 정도로 단순합니다. 그 결과는 10년 동안 NLP를 재구성했습니다.

## 개념

![Skip-gram 윈도우와 임베딩 공간](./assets/word2vec.svg)

**분포적 가설**(Firth, 1957): "단어의 의미는 그 단어를 둘러싼 환경에 의해 알 수 있다." 두 단어가 유사한 문맥에서 등장한다면, 그 단어들은 아마도 유사한 의미를 가질 것이다.

Word2Vec은 두 가지 형태로 제공되며, 둘 다 이 아이디어를 활용한다.

- **Skip-gram.** 중심 단어가 주어졌을 때 주변 단어를 예측한다. `cat -> (the, sat, on)` (윈도우 크기 2).
- **CBOW(연속적인 단어 주머니).** 주변 단어가 주어졌을 때 중심 단어를 예측한다. `(the, sat, on) -> cat`.

Skip-gram은 학습 속도가 느리지만 희귀 단어를 더 잘 처리한다. 이후 기본 방법으로 자리 잡았다.

네트워크는 비선형성이 없는 하나의 은닉층을 가진다. 입력은 어휘 집합에 대한 원-핫 벡터이다. 출력은 어휘 집합에 대한 소프트맥스이다. 학습 후 출력층은 버린다. 은닉층 가중치가 임베딩이다.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          이것이 임베딩이다
```

핵심 기법: 100k 단어에 대한 소프트맥스는 계산 비용이 너무 높다. Word2Vec은 **네거티브 샘플링**을 사용하여 이진 분류 작업으로 변환한다. "이 문맥 단어가 이 중심 단어 근처에 나타났는가, 예/아니오"를 예측한다. 전체 어휘 집합에 대한 소프트맥스 계산 대신, 각 학습 쌍에 대해 소수의 네거티브(공동 등장하지 않는) 단어를 샘플링한다.

## 구축 방법

### 1단계: 코퍼스에서 훈련 쌍 생성

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

윈도우 내의 모든 (중심, 문맥) 쌍은 양성 훈련 예시입니다.

### 2단계: 임베딩 테이블

두 개의 행렬. `W`는 중심 단어 임베딩 테이블(보존하는 것). `W'`는 문맥 단어 테이블(종종 폐기, 때로는 `W`와 평균화).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

작은 무작위 초기화. 어휘 크기 10k와 차원 100은 현실적; 교육용으로는 50 어휘 x 16 차원으로도 기하학적 구조를 볼 수 있습니다.

### 3단계: 네거티브 샘플링 목적 함수

각 양성 쌍 `(중심, 문맥)`에 대해 어휘에서 `k`개의 무작위 단어를 네거티브로 샘플링. `W[중심] · W'[문맥]`의 내적이 양성에서는 높고 네거티브에서는 낮도록 모델 훈련.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

마법 공식: 양성 쌍에 대한 로지스틱 손실(시그모이드 1에 가깝게) + 네거티브 쌍에 대한 로지스틱 손실(시그모이드 0에 가깝게). 그래디언트는 두 테이블로 흐름. 전체 유도는 원본 논문에 있음; 확실히 이해하려면 종이에 한 번 따라해 보세요.

### 4단계: 장난감 코퍼스 훈련

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

대규모 코퍼스에서 충분한 에포크 후, 문맥을 공유하는 단어는 유사한 중심 임베딩을 가짐. 장난감 코퍼스에서는 희미하게, 수십억 토큰에서는 극적으로 나타남.

### 5단계: 유추 트릭

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

사전 훈련된 300차원 Google News 벡터에서:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. 모델이 왕족을 알기 때문이 아님. 벡터 `(king - man)`이 "왕족"과 같은 것을 포착하고, 이를 `woman`에 더하면 왕족-여성 영역 근처에 도달하기 때문입니다.

## 사용 방법

Word2Vec을 처음부터 구현하는 것은 교육용입니다. 실제 프로덕션 NLP에서는 `gensim`을 사용합니다.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

실제 작업에서는 Word2Vec을 직접 학습시키는 경우가 거의 없습니다. 사전 학습된 벡터를 다운로드하여 사용합니다.

- **GloVe** — 스탠포드 대학의 동시 발생 행렬 분해 접근법. 50d, 100d, 200d, 300d 체크포인트. 일반적인 커버리지가 우수함. 레슨 04에서 GloVe를 구체적으로 다룹니다.
- **fastText** — 페이스북의 Word2Vec 확장 버전. 문자 n-그램을 임베딩하여 미등록 단어를 서브워드 조합으로 처리. 레슨 04.
- **Google News 사전 학습 Word2Vec** — 300d, 300만 단어 어휘, 2013년 공개. 여전히 매일 다운로드됨.

### 2026년에도 Word2Vec이 여전히 유리한 경우

- 경량 도메인 특화 검색. 노트북에서 1시간 동안 의학 초록을 학습시켜 일반 모델이 포착하지 못하는 특수 벡터 생성.
- 유추 스타일 특징 공학. `gender_vector = mean(man - woman 쌍)`. 다른 단어에서 이를 빼서 성별 중립 축 생성. 공정성 연구에서 여전히 사용됨.
- 해석 가능성. 100d는 PCA 또는 t-SNE로 플롯하기에 충분히 작아 실제 클러스터 형성을 확인할 수 있음.
- GPU 없이 장치에서 추론을 실행해야 하는 모든 경우. Word2Vec 조회는 단일 행 페치 작업.

### Word2Vec의 한계

다의어 문제. `bank`는 하나의 벡터만 가짐. `river bank`와 `financial bank`가 이를 공유. `table`(스프레드시트 vs. 가구)도 공유. 다운스트림 분류기는 벡터에서 의미 차이를 구분할 수 없음.

컨텍스트 임베딩(ELMo, BERT, 이후 모든 트랜스포머)은 주변 컨텍스트에 따라 단어 발생마다 다른 벡터를 생성함으로써 이 문제를 해결했습니다. 이것이 Word2Vec에서 BERT로의 도약: 정적에서 컨텍스트 임베딩으로의 전환. 7단계에서 트랜스포머 부분을 다룹니다.

미등록 단어 문제는 또 다른 한계. Word2Vec은 학습 데이터에 없는 `Zoomer-approved`를 본 적 없음. 대체 방법 없음. fastText는 서브워드 조합으로 이 문제를 해결(레슨 04).

## Ship It

저장 위치: `outputs/skill-embedding-probe.md`

```markdown
---
name: embedding-probe
description: word2vec 모델을 검사합니다. 유추 실행, 이웃 찾기, 품질 진단을 수행합니다.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, 임베딩, 디버깅]
---

학습된 단어 임베딩이 제대로 작동하는지 확인하기 위해 프로브를 실행합니다. `gensim.models.KeyedVectors` 객체와 어휘 사전이 주어졌을 때 다음 작업을 수행합니다:

1. 세 가지 표준 유추 테스트 실행. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. 상위 1개 결과와 코사인 유사도를 보고합니다.
2. 사용자가 제공한 도메인 특화 단어 5개에 대한 최근접 이웃 테스트. 코사인 유사도와 함께 상위 5개 이웃을 출력합니다.
3. 하나의 대칭성 검사. `similarity(a, b) == similarity(b, a)`가 부동소수점 정밀도 범위 내에서 성립하는지 확인합니다.
4. 하나의 퇴화 검사. 임베딩의 노름이 0.01 미만이거나 100을 초과하면 모델에 학습 버그가 있는 것으로 간주하고 플래그를 표시합니다.

유추 정확도만으로 모델을 양호하다고 선언하지 마십시오. 유추 벤치마크는 조작 가능하며 다운스트림 작업으로 전이되지 않습니다. 내재적 평가(intrinsic evaluation)와 다운스트림 평가를 함께 권장합니다.
```

## 연습 문제

1. **쉬움.** 작은 코퍼스(고양이와 개에 관한 20개 문장)로 훈련 루프를 실행합니다. 200 에포크 후, `nearest(vocab, W, W[vocab["cat"]])`가 상위 3개 결과에 `dog`를 반환하는지 확인합니다. 만약 그렇지 않다면 에포크 수나 어휘 크기를 늘립니다.
2. **중간.** 빈도가 높은 단어의 서브샘플링을 추가합니다. 빈도가 `10^-5`를 초과하는 단어는 빈도에 비례하는 확률로 훈련 쌍에서 제외합니다. 이 방법이 희귀 단어 유사도에 미치는 영향을 측정합니다.
3. **어려움.** 20 Newsgroups 코퍼스로 모델을 훈련시킵니다. 두 개의 편향 축인 `he - she`와 `doctor - nurse`를 계산합니다. 직업 단어들을 두 축에 투영합니다. 가장 큰 편향 차이를 보이는 직업들을 보고합니다. 이는 공정성 연구자들이 사용하는 탐색 방법과 동일합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 단어 임베딩(Word embedding) | 단어를 벡터로 | 문맥에서 학습된 밀집된 저차원(일반적으로 100-300) 표현 |
| 스킵-그램(Skip-gram) | Word2Vec 트릭 | 중심 단어에서 문맥 단어 예측. CBOW보다 느리지만 희귀 단어에 더 적합 |
| 네거티브 샘플링(Negative sampling) | 훈련 단축 기법 | 전체 어휘 집합에 대한 소프트맥스 대신 `k`개의 무작위 단어에 대한 이진 분류로 대체 |
| 정적 임베딩(Static embedding) | 단어당 하나의 벡터 | 문맥과 무관하게 동일한 벡터. 다의어 처리 실패 |
| 문맥적 임베딩(Contextual embedding) | 문맥 감지 벡터 | 주변 단어에 따라 각 발생마다 다른 벡터. 트랜스포머가 생성하는 것 |
| OOV(Out of vocabulary) | 훈련 시 미등장 단어 | 훈련 중 보지 못한 단어. Word2Vec은 이에 대한 벡터 생성 불가 |

## 추가 자료

- [Mikolov et al. (2013). 단어 및 구의 분산 표현과 그 구성성](https://arxiv.org/abs/1310.4546) — 네거티브 샘플링 논문. 짧고 읽기 쉬움.
- [Rong, X. (2014). word2vec 파라미터 학습 설명](https://arxiv.org/abs/1411.2738) — 원본 논문의 수학이 어렵게 느껴질 경우, 가장 명확한 그래디언트 유도 설명.
- [gensim Word2Vec 튜토리얼](https://radimrehurek.com/gensim/models/word2vec.html) — 실제로 작동하는 프로덕션 학습 설정.