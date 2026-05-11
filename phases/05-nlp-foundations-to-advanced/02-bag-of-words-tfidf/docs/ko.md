# Bag of Words, TF-IDF, 그리고 텍스트 표현

> 먼저 세고, 나중에 생각하라. TF-IDF는 2026년에도 잘 정의된 작업에서 임베딩을 능가한다.

**유형:** 구축
**언어:** Python
**선수 지식:** Phase 5 · 01 (텍스트 처리), Phase 2 · 02 (선형 회귀 기초부터)
**소요 시간:** ~75분

## 문제 정의

모델은 숫자를 필요로 합니다. 하지만 우리는 문자열을 가지고 있습니다.

모든 NLP 파이프라인은 동일한 질문에 답해야 합니다. 가변 길이 토큰 스트림을 분류기가 처리할 수 있는 고정 크기 벡터로 어떻게 변환할까요? 이 분야에서 처음 제시한 해결책은 가장 단순하지만 효과적인 방법이었습니다. 단어를 세는 것입니다. 벡터를 만드는 것입니다.

이 벡터는 어떤 임베딩 모델보다 더 많은 프로덕션 NLP 시스템을 지탱해 왔습니다. 스팸 필터, 주제 분류기, 로그 이상 감지, 검색 순위 지정(BM25 이전), 초기 감성 분석, 학계 NLP 벤치마크의 첫 10년 등. 2026년에도 여전히 많은 실무자들은 좁은 분류 작업에서 이 방법을 먼저 선택합니다. 이 방법은 빠르고 해석이 용이하며, 단어 존재 여부가 중요한 작업에서는 4억 개 파라미터 임베딩 모델과 구분하기 어려울 정도로 좋은 성능을 보입니다.

이 강의에서는 단어 가방(Bag of Words) 모델을 직접 구현한 후 TF-IDF를 처음부터 만들어 봅니다. 그런 다음 scikit-learn을 사용해 동일한 작업을 단 3줄로 수행하는 방법을 보여줍니다. 마지막으로 임베딩을 필요로 하게 만드는 실패 사례를 설명합니다.

## 개념

![BoW vs TF-IDF 표현 흐름](./assets/bow-tfidf.svg)

**Bag of Words (BoW)**는 단어 순서를 무시합니다. 각 문서에서 어휘(vocabulary)에 있는 각 단어가 몇 번 등장하는지 카운트합니다. 벡터 길이는 어휘(vocabulary) 크기이며, 위치 `i`는 단어 `i`의 등장 횟수입니다.

**TF-IDF**는 BoW에 가중치를 재할당합니다. 모든 문서에 등장하는 단어는 정보성이 낮으므로 가중치를 줄이고, 코퍼스(corpus) 전체에서는 드물지만 특정 문서에 자주 등장하는 단어는 신호(signal)이므로 가중치를 높입니다.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

여기서 `TF`는 문서 내 단어 빈도(term frequency), `df`는 문서 빈도(document frequency, 단어를 포함하는 문서 수), `N`은 전체 문서 수입니다. `log`는 널리 사용되는 단어에 대한 가중치를 제한합니다.

주요 특징: 둘 다 해석 가능한 축(interpretable axes)을 가진 희소 벡터(sparse vectors)를 생성합니다. 학습된 분류기의 가중치를 보고 어떤 단어가 문서를 각 클래스로 분류하는지 확인할 수 있습니다. 768차원 BERT 임베딩에서는 이 작업을 수행할 수 없습니다.

## 구축 단계

### 1단계: 어휘 구축

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

입력: 토큰화된 문서 목록(어떤 단어 수준 토크나이저도 가능; 이 레슨의 `code/main.py`는 단순화된 소문자 변형 사용). 출력: `{단어: 인덱스}` 딕셔너리. 안정적인 삽입 순서는 첫 번째 문서에서 처음 본 단어가 인덱스 0이 됨을 의미합니다. 관례는 다양함; scikit-learn은 알파벳 순으로 정렬합니다.

### 2단계: 단어 가방(Bag of Words)

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

행은 문서입니다. 열은 어휘 인덱스입니다. 항목 `[i][j]`는 "문서 `i`에서 단어 `j`가 몇 번 나타나는지"를 나타냅니다. 문서 1은 `cat`이 두 번 나타납니다. 문서 0은 `ran`이 0번 나타납니다.

### 3단계: 단어 빈도(TF)와 문서 빈도(DF)

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

명명할 가치가 있는 두 가지 스무딩 기법. `(n+1)/(d+1)`은 `log(x/0)`를 방지합니다. 마지막 `+1`은 모든 문서에 나타나는 단어의 IDF가 0이 아닌 1이 되도록 하여 scikit-learn의 기본값과 일치시킵니다. 다른 구현에서는 `log(N/df)`를 그대로 사용합니다. 둘 다 작동하지만 스무딩된 버전이 더 친절합니다.

### 4단계: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

3개의 문서, 5개의 어휘 단어(`the`, `cat`, `sat`, `dog`, `ran`). `the`는 3개 문서 모두에 나타나므로 IDF가 낮습니다. `dog`는 1개 문서에만 나타나므로 IDF가 높습니다. 벡터는 희소하며(대부분 항목이 작음) 판별 단어가 두드러집니다.

### 5단계: 행 L2 정규화

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

정규화 없이는 긴 문서가 더 큰 벡터를 얻어 유사도 점수를 지배합니다. L2 정규화는 모든 문서를 단위 초구면에 놓습니다. 행 간 코사인 유사도는 이제 단순히 내적과 같습니다.

## 사용 방법

scikit-learn은 프로덕션 버전을 제공합니다.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer`는 토크나이징, 어휘 사전 생성, BoW(Bag of Words)를 한 번의 호출로 수행합니다. `TfidfVectorizer`는 IDF 가중치 및 L2 정규화를 추가합니다. 둘 다 희소 행렬을 반환합니다. 100k 문서의 경우 밀집 버전은 메모리에 맞지 않으므로 분류기가 밀집 행렬을 요구할 때까지 희소 행렬을 유지합니다.

모든 것을 바꾸는 주요 매개변수:

| 매개변수 | 효과 |
|-----|--------|
| `ngram_range=(1, 2)` | 바이그램 포함. 일반적으로 분류 성능 향상. |
| `min_df=2` | 2개 미만의 문서에 등장하는 단어 제거. 노이즈가 많은 데이터에서 어휘 사전 축소. |
| `max_df=0.95` | 95% 이상의 문서에 등장하는 단어 제거. 하드코딩된 목록 없이 불용어 제거 효과. |
| `stop_words="english"` | scikit-learn의 내장 불용어 목록. 작업에 따라 다름 — 감성 분석은 부정어(`not`)를 제거하면 안 됨. |
| `sublinear_tf=True` | 원시 `tf` 대신 `1 + log(tf)` 사용. 한 문서에서 용어가 반복될 때 유용. |

### TF-IDF가 여전히 우월한 경우 (2026년 기준)

- 스팸 탐지, 주제 라벨링, 로그 이상 감지. 단어 존재 여부가 중요; 의미적 뉘앙스는 불필요.
- 데이터 부족 환경(수백 개의 레이블된 예제). TF-IDF + 로지스틱 회귀는 사전 학습 비용이 없음.
- 지연 시간이 중요한 모든 경우. TF-IDF + 선형 모델은 마이크로초 단위로 응답. 트랜스포머를 통한 문서 임베딩은 10-100ms 소요.
- 예측 결과를 설명해야 하는 시스템. 분류기의 계수를 검사. 상위 긍정 단어가 이유.

### TF-IDF가 실패하는 경우

의미적 맹목성 실패. 다음 두 문서를 고려:

- "The movie was not good at all."
- "The movie was excellent."

하나는 부정적 리뷰, 다른 하나는 긍정적 리뷰입니다. 이들의 TF-IDF 교집합은 정확히 `{the, movie, was}`입니다. BoW 분류기는 `not`이 `good` 근처에 있을 때 레이블이 반전된다는 것을 암기해야 합니다. 충분한 데이터로는 학습할 수 있지만, 구문을 이해하는 모델만큼 우아하지는 않습니다.

다른 실패 사례: 추론 시 어휘 사전 외 단어. IMDb 리뷰로 학습된 BoW 모델은 훈련 시 등장하지 않은 `Zoomer-approved`를 처리할 수 없습니다. 서브워드 임베딩(레슨 04)은 이를 처리하지만 TF-IDF는 불가능합니다.

### 하이브리드: TF-IDF 가중치 임베딩

2026년 중간 규모 분류 작업의 실용적 기본값: TF-IDF 가중치를 단어 임베딩에 대한 어텐션으로 사용.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

임베딩에서 의미적 용량을, TF-IDF에서 희귀 단어 강조를 얻습니다. 분류기는 풀링된 벡터로 학습됩니다. 이는 약 50k 레이블 예제 미만의 감성, 주제, 의도 분류에서 단독 사용보다 성능이 우수합니다.

## Ship It

`outputs/prompt-vectorization-picker.md`로 저장:

```markdown
---
name: vectorization-picker
description: 텍스트 분류 작업이 주어졌을 때, BoW, TF-IDF, 임베딩 또는 하이브리드 방식을 추천합니다.
phase: 5
lesson: 02
---

텍스트 벡터화 전략을 추천합니다. 작업 설명이 주어지면 다음을 출력합니다:

1. 표현 방식 (BoW, TF-IDF, 트랜스포머 임베딩, 또는 하이브리드). 한 문장으로 이유를 설명합니다.
2. 구체적인 벡터라이저 설정. 라이브러리 이름을 명시합니다. 인수(`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`)를 인용합니다.
3. 배포 전 테스트해야 할 하나의 실패 모드.

라벨이 지정된 예시가 500개 미만일 때 임베딩을 추천하지 않습니다. 단, TF-IDF 베이스라인에서 의미론적 실패 증거가 있는 경우는 예외입니다. 감성 분석에서는 불용어 제거를 거부합니다(부정어가 신호를 전달함). 클래스 불균형은 벡터라이저 변경 이상의 조치가 필요하다고 표시합니다.

예시 입력: "30k개의 고객 지원 티켓을 12개 카테고리로 분류합니다. 대부분의 티켓은 2-3문장입니다. 영어만 사용. 감사 로그를 위한 설명 가능성이 필요합니다."

예시 출력:

- 표현 방식: TF-IDF. 30k 예시는 작지 않으며, 설명 가능성 요구 사항으로 인해 밀집 임베딩은 제외됩니다.
- 설정: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. 카테고리 키워드가 때때로 불용어일 수 있으므로 불용어 유지("not working" vs "working").
- 테스트 실패 모드: `min_df=3`이 희귀 카테고리 키워드를 제거하지 않는지 확인합니다. 클래스별로 필터링된 `get_feature_names_out`을 실행하고 육안으로 점검합니다.
```

## 연습 문제

1. **쉬움.** L2-정규화된 TF-IDF 출력에 대해 `cosine_similarity(doc_vec_a, doc_vec_b)`를 구현하세요. 동일한 문서는 1.0 점수를, 어휘가 전혀 겹치지 않는 문서는 0.0 점수를 받는지 검증하세요.
2. **중간.** `bag_of_words`에 `n-gram` 지원을 추가하세요. 매개변수 `n`은 `n-gram`에 대한 카운트를 생성합니다. `n=2`일 때 `["the", "cat", "sat"]`에 대해 `["the cat", "cat sat"]`의 바이그램 카운트가 생성되는지 테스트하세요.
3. **어려움.** GloVe 100d 벡터(1회 다운로드 후 캐시)를 사용하여 위의 TF-IDF-가중-임베딩 하이브리드를 구축하세요. 20 Newsgroups 데이터셋에서 일반 TF-IDF 및 일반 평균 풀링 임베딩과 분류 정확도를 비교하세요. 어떤 방법이 어떤 경우에 더 우수한지 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| BoW | 단어 빈도 벡터 | 한 문서 내 어휘 단어들의 개수. 순서는 무시. |
| TF | 용어 빈도 | 문서 내 단어 등장 횟수. 문서 길이로 정규화 가능. |
| DF | 문서 빈도 | 단어가 최소 한 번 이상 등장하는 문서의 개수. |
| IDF | 역문서 빈도 | `log(N / df)`로 평활화. 모든 문서에 등장하는 단어를 가중치 감소. |
| 희소 벡터 | 대부분 0 | 어휘는 일반적으로 10k-100k 단어; 대부분의 단어는 특정 문서에 없음. |
| 코사인 유사도 | 벡터 각도 | L2-정규화된 벡터의 내적. 1은 동일, 0은 직교. |

## 추가 자료

- [scikit-learn — 텍스트에서 특징 추출](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 표준 API 참조, 모든 조정 가능한 파라미터에 대한 설명 포함.
- [Salton, G., & Buckley, C. (1988). 자동 텍스트 검색에서의 용어 가중치 접근법](https://www.sciencedirect.com/science/article/pii/0306457388900210) — TF-IDF를 10년 간 기본 방법으로 만든 논문.
- ["TF-IDF가 여전히 임베딩을 능가하는 이유" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026년 관점에서 구식 방법이 승리하는 경우와 그 이유.