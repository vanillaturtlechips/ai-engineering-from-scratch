# 토픽 모델링 — LDA와 BERTopic

> LDA: 문서는 토픽의 혼합체, 토픽은 단어들의 분포. BERTopic: 문서는 임베딩 공간에서 클러스터링, 클러스터는 토픽. 같은 목표, 다른 기본 요소.

**유형:** 학습
**언어:** Python
**사전 요구 지식:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**소요 시간:** ~45분

## 문제 정의

10,000개의 고객 지원 티켓, 50,000개의 뉴스 기사, 또는 200,000개의 트윗이 있습니다. 이를 직접 읽지 않고도 전체 컬렉션이 어떤 내용인지 알아야 합니다. 레이블이 지정된 카테고리도 없고, 몇 개의 카테고리가 존재하는지도 모릅니다.

토픽 모델링은 이러한 문제를 비지도 방식으로 해결합니다. 말뭉치(corpus)를 입력하면, 일관성 있는 소수의 토픽과 각 문서에 대한 해당 토픽들의 분포를 반환합니다.

두 가지 알고리즘 계열이 주로 사용됩니다. LDA(2003)는 각 문서를 잠재 토픽들의 혼합으로, 각 토픽을 단어들의 분포로 취급합니다. 추론은 베이지안 방식으로 이루어집니다. 혼합 멤버십 토픽 할당과 설명 가능한 단어 수준 확률 분포가 필요한 프로덕션 환경에서 여전히 사용됩니다.

BERTopic(2020)은 BERT로 문서를 인코딩하고, UMAP으로 차원을 축소한 후 HDBSCAN으로 클러스터링하며, 클래스 기반 TF-IDF를 통해 토픽 단어들을 추출합니다. 짧은 텍스트, 소셜 미디어, 단어 중복보다 의미적 유사성이 중요한 모든 경우에 우수한 성능을 보입니다. 단, 하나의 문서에 하나의 토픽만 할당되는 것은 장문 콘텐츠에 대한 한계입니다.

이 강의에서는 두 방법에 대한 직관을 구축하고, 주어진 말뭉치에 어떤 방법을 선택할지 결정하는 방법을 설명합니다.

## 개념

![LDA 혼합 모델 vs BERTopic 클러스터링](../assets/topic-modeling.svg)

**LDA 생성적 이야기.** 각 토픽은 단어들의 분포입니다. 각 문서는 토픽들의 혼합입니다. 문서 내 단어를 생성하려면 문서의 혼합에서 토픽을 샘플링한 후, 해당 토픽의 분포에서 단어를 샘플링합니다. 추론은 이를 역으로 수행합니다: 관측된 단어들을 바탕으로 문서별 토픽 분포와 토픽별 단어 분포를 추론합니다. Collapsed Gibbs 샘플링 또는 변분 베이지안이 수학적 계산을 수행합니다.

주요 LDA 출력:

- `doc_topic`: 행렬 `(n_docs, n_topics)`, 각 행의 합은 1 (문서의 토픽 혼합).
- `topic_word`: 행렬 `(n_topics, vocab_size)`, 각 행의 합은 1 (토픽의 단어 분포).

**BERTopic 파이프라인.**

1. 각 문서를 문장 임베더(예: `all-MiniLM-L6-v2`)로 인코딩합니다. 384차원 벡터입니다.
2. UMAP으로 차원을 ~5차원으로 축소합니다. BERT 임베딩은 클러스터링에 너무 고차원이기 때문입니다.
3. HDBSCAN으로 클러스터링합니다. 밀도 기반이며, 가변 크기 클러스터와 "이상치" 레이블을 생성합니다.
4. 각 클러스터에 대해 클러스터 내 문서들을 기반으로 클래스 기반 TF-IDF를 계산하여 상위 단어들을 추출합니다.

출력은 문서당 하나의 토픽(이상치 레이블 -1 포함)입니다. 선택적으로 HDBSCAN의 확률 벡터를 통해 소프트 멤버십을 제공할 수 있습니다.

## 구축 방법

### 1단계: scikit-learn을 통한 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

주의: 불용어 제거, min_df 및 max_df는 희귀 단어와 빈번한 단어를 필터링, LDA는 원시 카운트를 요구하므로 TfidfVectorizer가 아닌 CountVectorizer 사용.

### 2단계: BERTopic (프로덕션)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 필터는 BERTopic의 이상치 버킷(HDBSCAN이 클러스터링하지 못한 문서)을 제거합니다. `min_topic_size`는 HDBSCAN의 최소 클러스터 크기를 제어하며, BERTopic 라이브러리 기본값은 10입니다. 이 예제에서는 레슨 규모에 맞춰 15로 명시적으로 설정했습니다. 10,000개 이상의 문서 코퍼스의 경우 50 또는 100으로 증가시키세요.

### 3단계: 평가

두 방법 모두 토픽 단어를 출력합니다. 문제는 해당 단어들이 일관성을 갖는지 여부입니다.

- **토픽 일관성(c_v).** 슬라이딩 윈도우 컨텍스트에서 상위 단어 쌍의 NPMI(Normalized Pointwise Mutual Information)를 결합하고, 점수를 토픽 벡터로 집계한 후 코사인 유사도를 통해 해당 벡터를 비교합니다. 높을수록 좋습니다. `gensim.models.CoherenceModel`을 `coherence="c_v"`와 함께 사용하세요.
- **토픽 다양성.** 모든 토픽의 상위 단어에서 고유 단어의 비율. 높을수록 좋습니다(토픽 간 중복 없음).
- **정성적 검사.** 각 토픽의 상위 단어를 읽어보세요. 실제 사물을 나타내는가? 인간의 판단은 여전히 마지막 방어선입니다.

## 어떤 것을 선택할지

| 상황 | 선택 |
|-----------|------|
| 짧은 텍스트 (트윗, 리뷰, 헤드라인) | BERTopic |
| 주제 혼합이 있는 긴 문서 | LDA |
| GPU 없음 / 제한된 계산 자원 | LDA 또는 NMF |
| 문서 수준의 다중 주제 분포 필요 | LDA |
| 주제 라벨링을 위한 LLM 통합 | BERTopic (직접 지원) |
| 리소스가 제한된 엣지 배포 | LDA |
| 최대 의미적 일관성 | BERTopic |

가장 큰 실용적 고려사항은 문서 길이입니다. BERT 임베딩은 잘리며, LDA 카운팅은 어떤 길이든 처리 가능합니다. 임베딩 모델의 컨텍스트보다 긴 문서의 경우, 청크화 + 집계하거나 LDA를 사용해야 합니다.

## 사용 방법

2026 스택:

- **BERTopic.** 짧은 텍스트 및 의미론적 요소가 중요한 모든 경우의 기본 선택.
- **`gensim.models.LdaModel`.** 프로덕션용 클래식 LDA, 성숙하고 검증된 모델.
- **`sklearn.decomposition.LatentDirichletAllocation`.** 실험용 쉬운 LDA.
- **NMF.** 비음수 행렬 분해. LDA에 대한 빠른 대안, 짧은 텍스트에서 비슷한 품질.
- **Top2Vec.** BERTopic과 유사한 설계. 커뮤니티는 작지만 일부 벤치마크에서 우수.
- **FASTopic.** 최신 기술, 매우 큰 코퍼스에서 BERTopic보다 빠름.
- **LLM 기반 라벨링.** 어떤 클러스터링이든 실행한 후, 모델에 각 클러스터 이름 지정 요청.

## Ship It

`outputs/skill-topic-picker.md`로 저장:

```markdown
---
name: topic-picker
description: 코퍼스에 대해 LDA 또는 BERTopic을 선택합니다. 라이브러리, 설정, 평가를 명시합니다.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, 토픽-모델링]
---

코퍼스 설명(문서 수, 평균 길이, 도메인, 언어, 계산 예산)이 주어졌을 때 출력:

1. 알고리즘. LDA / NMF / BERTopic / Top2Vec / FASTopic. 한 문장 이유.
2. 설정. 토픽 수: `recommended = max(5, round(sqrt(n_docs)))`, 40,000개 미만 문서 코퍼스는 200으로 제한; 코퍼스가 매우 큰 경우(>40k)에만 200 초과 허용하며 계산 비용 증가를 명시. `min_df` / `max_df` 필터와 신경망 접근법의 임베딩 모델도 포함.
3. 평가. `gensim.models.CoherenceModel`을 통한 토픽 일관성(c_v), 토픽 다양성, 20개 샘플 인간 평가.
4. 실패 모드 검증. LDA의 경우 불용어와 빈도 높은 단어를 흡수하는 "쓰레기 토픽". BERTopic의 경우 모호한 문서를 삼키는 -1 이상치 클러스터.

임베딩 모델의 컨텍스트 윈도우보다 긴 문서에 대해 청킹 전략 없이 BERTopic을 거부합니다. 10토큰 미만 매우 짧은 텍스트(트윗, 리뷰)에 대해 일관성이 붕괴되므로 LDA를 거부합니다. 5 미만 n_topics 선택은 잘못된 것으로 플래그; 40,000개 미만 문서 코퍼스에서 200 초과는 과도한 분할로 플래그.
```

## 연습 문제

1. **쉬움.** 20 Newsgroups 데이터셋에 5개의 토픽을 갖는 LDA를 적합하세요. 각 토픽별 상위 10개 단어를 출력하세요. 각 토픽에 수동으로 레이블을 지정하세요. 알고리즘이 실제 카테고리를 찾았나요?
2. **중간.** 동일한 20 Newsgroups 부분 집합에 BERTopic을 적합하세요. 발견된 토픽 수, 상위 단어, LDA 대비 정성적 일관성(coherence)을 비교하세요. 어떤 방법이 실제 카테고리를 더 명확하게 표현하나요?
3. **어려움.** LDA와 BERTopic 모두에 대해 코퍼스의 c_v 일관성(coherence)을 계산하세요. 각각 5, 10, 20, 50개의 토픽으로 실행하세요. 토픽 수에 따른 일관성(coherence) 변화를 플롯하세요. 어떤 방법이 토픽 수에 걸쳐 더 안정적인지 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 토픽(Topic) | 코퍼스가 다루는 주제 | 단어들의 확률 분포(LDA) 또는 유사한 문서들의 클러스터(BERTopic) |
| 혼합 멤버십(Mixed membership) | 문서가 여러 토픽을 가짐 | LDA는 각 문서에 모든 토픽에 대한 분포를 할당함 |
| UMAP(Uniform Manifold Approximation and Projection) | 차원 축소 | 지역 구조를 보존하는 매니폴드 학습; BERTopic에서 사용됨 |
| HDBSCAN(Hierarchical Density-Based Spatial Clustering of Applications with Noise) | 밀도 기반 클러스터링 | 가변 크기 클러스터를 찾음; 이상치에 대해 "노이즈" 라벨(-1)을 생성함 |
| c_v 일관성(Coherence) | 토픽 품질 지표 | 슬라이딩 윈도우 내 상위 토픽 단어들의 평균 점별 상호 정보량 |

## 추가 자료

- [Blei, Ng, Jordan (2003). 잠재 디리클레 할당(Latent Dirichlet Allocation)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 논문.
- [Grootendorst (2022). BERTopic: 클래스 기반 TF-IDF 절차를 활용한 신경망 기반 토픽 모델링](https://arxiv.org/abs/2203.05794) — BERTopic 논문.
- [Röder, Both, Hinneburg (2015). 토픽 일관성 측정 공간 탐색](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — c_v 및 관련 지표를 소개한 논문.
- [BERTopic 문서](https://maartengr.github.io/BERTopic/) — 프로덕션 레퍼런스. 우수한 예제 포함.