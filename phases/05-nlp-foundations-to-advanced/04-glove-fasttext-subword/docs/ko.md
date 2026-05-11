# GloVe, FastText, 그리고 서브워드 임베딩

> Word2Vec은 단어당 하나의 임베딩을 학습했습니다. GloVe는 공분산 행렬을 분해했습니다. FastText는 조각들을 임베딩했습니다. BPE는 트랜스포머로 가는 다리를 놓았습니다.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** 5단계 · 03 (Word2Vec 처음부터 구현)
**소요 시간:** ~45분

## 문제 정의

Word2Vec은 두 가지 미해결 문제를 남겼습니다.

첫째, 온라인 스킵-그램 업데이트 대신 공분산 행렬을 직접 분해(LSA, HAL)하는 병렬 연구 라인이 있었습니다. Word2Vec의 반복적 접근이 근본적으로 더 나은 것이었는지, 아니면 두 방법이 카운트를 처리하는 방식의 차이에서 비롯된 것인지 **GloVe**가 답했습니다: 신중하게 선택된 손실 함수를 사용한 행렬 분해는 Word2Vec과 동등하거나 더 나은 성능을 보이며, 학습 비용도 더 적게 듭니다.

둘째, 두 방법 모두 본 적 없는 단어에 대한 해결책이 없었습니다. `Zoomer-approved`, `dogecoin`, 지난주에 만들어진 모든 고유 명사, 희귀 어근의 모든 활용형 등. **FastText**는 문자 n-그램 임베딩으로 이를 해결했습니다: 단어는 형태소를 포함한 부분들의 합이므로, 어휘집에 없는 단어도 의미 있는 벡터를 얻을 수 있습니다.

셋째, 트랜스포머가 등장하자 문제가 다시 바뀌었습니다. 단어 수준 어휘집은 약 백만 개 항목으로 제한되지만, 실제 언어는 그보다 더 개방적입니다. **바이트-페어 인코딩(BPE)**과 그 변형들은 모든 것을 포괄하는 빈번한 서브워드 단위 어휘집을 학습함으로써 이 문제를 해결했습니다. 모든 현대 LLM의 토크나이저는 서브워드 토크나이저입니다.

이 강의에서는 이 세 가지 방법을 모두 설명한 후, 어떤 상황에서 어떤 방법을 선택해야 하는지 설명합니다.

## 개념

![세 가지 임베딩 접근법: GloVe 공동 발생, FastText 서브워드, BPE 병합](./assets/embeddings.svg)

**GloVe (Global Vectors).** 단어-단어 공동 발생 행렬 `X`를 구축합니다. 여기서 `X[i][j]`는 단어 `i`의 컨텍스트에서 단어 `j`가 나타나는 빈도입니다. 벡터 `v_i`와 `v_j`의 내적에 바이어스 `b_i`와 `b_j`를 더한 값이 `log(X[i][j])`에 가깝도록 학습합니다. 손실 함수에 가중치를 적용하여 빈번한 단어 쌍이 학습을 지배하지 않도록 합니다. 완료.

**FastText.** 단어는 문자 n-그램들의 합에 단어 자체를 더한 것입니다. 예를 들어 `where`는 `<wh, whe, her, ere, re>, <where>`로 표현됩니다. 단어 벡터는 이러한 구성 요소 벡터들의 합입니다. Word2Vec 방식으로 학습합니다. 장점: 보이지 않는 단어(예: `whereupon`)도 알려진 n-그램들로 조합될 수 있습니다.

**BPE (Byte-Pair Encoding).** 개별 바이트(또는 문자)로 구성된 어휘로 시작합니다. 코퍼스에서 모든 인접 쌍을 카운트합니다. 가장 빈번한 쌍을 새로운 토큰으로 병합합니다. 이 과정을 `k`번 반복합니다. 결과: `k + 256`개의 토큰으로 구성된 어휘가 생성됩니다. 빈번한 시퀀스(예: `ing`, `tion`, `the`)는 단일 토큰이 되고, 희귀한 단어는 익숙한 조각들로 분해됩니다. 모든 문장은 토큰화됩니다.

## 구축 방법

### GloVe: 공발생 행렬 분해

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

이름 붙일 만한 두 가지 이동 부분. 가중치 함수 `f(x) = (x/x_max)^alpha`는 매우 빈번한 쌍(예: `(the, and)`)을 손실 함수에서 지배하지 않도록 가중치를 낮춥니다. 최종 임베딩은 `W`(중심어)와 `W_tilde`(문맥) 테이블의 합입니다. 두 테이블을 합치는 것은 하나만 사용하는 것보다 성능이 좋은 공개된 기법입니다.

### FastText: 서브워드 인식 임베딩

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

각 단어는 n-그램 집합(일반적으로 3~6자)으로 표현됩니다. 단어 임베딩은 해당 n-그램 임베딩의 합입니다. Skip-gram 학습 시 Word2Vec에서 단일 벡터를 사용했던 부분에 이를 적용합니다.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

처음 보는 단어라도 일부 n-그램이 알려져 있다면 벡터를 얻을 수 있습니다. `whereupon`은 `<wh`, `her`, `ere`, `<where>`를 `where`과 공유하므로 두 단어는 서로 가까이 위치합니다.

### BPE: 학습된 서브워드 어휘

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

첫 번째 반복에서는 가장 흔한 인접 쌍을 병합합니다. 충분한 반복 후에는 빈번한 부분 문자열(`low`, `est`, `tion`)이 단일 토큰이 되고 희귀 단어는 깔끔하게 분할됩니다.

실제 GPT/BERT/T5 토크나이저는 30k~100k 병합을 학습합니다. 결과: 모든 텍스트는 알려진 ID의 유계 길이 시퀀스로 토큰화되며 OOV(Out-Of-Vocabulary)가 발생하지 않습니다.

## 사용 방법

실제로 이 중 어떤 것도 직접 훈련시키는 경우는 거의 없습니다. 사전 훈련된 체크포인트를 로드합니다.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

트랜스포머 시대의 BPE 스타일 서브워드 토크나이저:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 접두사는 단어 경계를 표시합니다(GPT-2 규칙). 모든 현대 토크나이저는 BPE 변형, WordPiece(BERT), 또는 SentencePiece(T5, LLaMA)입니다.

### 어떤 것을 선택할까

| 상황 | 선택 |
|-----------|------|
| 사전 훈련된 범용 단어 벡터, OOV 허용 필요 없음 | GloVe 300d |
| 사전 훈련된 범용 단어 벡터, 오타/신조어/형태론적으로 풍부한 언어 처리 필요 | FastText |
| 트랜스포머에 입력되는 모든 것(훈련 또는 추론) | 모델이 함께 제공된 토크나이저. 절대 교체하지 마세요. |
| 처음부터 직접 언어 모델 훈련 | 먼저 코퍼스에 BPE 또는 SentencePiece 토크나이저를 훈련시키세요 |
| 선형 모델을 사용한 프로덕션 텍스트 분류 | 여전히 TF-IDF. 레슨 02. |

## Ship It

`outputs/skill-tokenizer-picker.md`로 저장:

```markdown
---
name: tokenizer-picker
description: 새로운 언어 모델 또는 텍스트 파이프라인을 위한 토크나이저 접근 방식 선택
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, 토크나이저, 임베딩]
---

작업과 데이터셋 설명이 주어졌을 때, 다음을 출력합니다:

1. 토크나이저 전략(단어 수준, BPE, WordPiece, SentencePiece, 바이트 수준). 한 문장으로 이유 설명.
2. 목표 어휘 크기(예: 영어 전용 LM의 경우 32k, 다국어 경우 64k-100k).
3. 정확한 학습 명령어가 포함된 라이브러리 호출. 라이브러리 이름 명시. 인수 인용.
4. 재현성 문제 1가지. 토크나이저-모델 불일치는 가장 흔한 무음 프로덕션 버그입니다. 함께 사용해야 하는 쌍을 명시.

사전 학습된 LLM을 파인튜닝하는 경우 커스텀 토크나이저 학습 권장을 거부합니다. 프로덕션 추론을 목표로 하는 모든 모델에 단어 수준 토크나이저 권장을 거부합니다. 비영어/다중 문자 집합 코퍼스는 바이트 폴백이 있는 SentencePiece가 필요하다고 표시합니다.
```

## 연습 문제

1. **쉬움.** `char_ngrams("playing")`과 `char_ngrams("played")`를 실행하세요. 두 n-gram 집합의 자카드 유사도(Jaccard similarity)를 계산하세요. `pla`, `lay`, `play`와 같은 상당한 공유 조각이 나타날 것입니다. 이는 FastText가 형태소 변형 간에 잘 전이(transfer)되는 이유입니다.
2. **중간.** `learn_bpe`를 확장하여 어휘(vocabulary) 성장을 추적하세요. 병합(merge) 횟수에 따른 코퍼스 문자당 토큰 수를 그래프로 그리세요. 처음에는 빠른 압축이 일어나다가 ~2-3문자당 토큰 근처에서 점근할 것입니다.
3. **어려움.** 셰익스피어 전집에 1k-병합 BPE를 학습시키세요. 일반 단어와 희귀 고유명사의 토큰화를 비교하세요. 단어당 평균 토큰 수를 학습 전후 측정하세요. 어떤 결과가 놀라웠는지 작성하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| Co-occurrence matrix | 단어-단어 빈도 테이블 | `X[i][j]` = 단어 `i` 주변 윈도우에서 단어 `j`가 나타나는 빈도. |
| Subword | 단어 조각 | 문자 n-gram(FastText) 또는 학습된 토큰(BPE/WordPiece/SentencePiece). |
| BPE | Byte-pair encoding | 어휘 크기가 목표치에 도달할 때까지 가장 빈번한 인접 쌍을 반복적으로 병합. |
| OOV | Out of vocabulary | 모델이 본 적 없는 단어. Word2Vec/GloVe는 실패. FastText와 BPE는 처리 가능. |
| Byte-level BPE | 원시 바이트에 대한 BPE | GPT-2의 방식. 어휘는 256개 바이트로 시작하므로 OOV가 절대 발생하지 않음. |

## 추가 자료

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 논문, 7페이지, 손실 함수(loss function) 유도의 정석을 보여줌.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE(Byte-Pair Encoding)를 현대 NLP에 도입한 논문.
- [Hugging Face 토크나이저 요약](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE, WordPiece, SentencePiece가 실제 구현에서 어떻게 다른지 설명.