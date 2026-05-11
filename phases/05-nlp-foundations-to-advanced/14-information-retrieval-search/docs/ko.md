# 정보 검색과 검색

> BM25는 정확하지만 취약합니다. Dense는 넓은 범위를 커버하지만 키워드를 놓칩니다. Hybrid는 2026년 기본값입니다. 나머지는 튜닝입니다.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**소요 시간:** ~75분

## 문제 정의

사용자가 "what happens if someone lies to get money"를 입력하면 실제로 해당 행위를 규정하는 법령인 "Section 420 IPC"를 찾기를 기대합니다. 키워드 검색은 공유 어휘가 없어 완전히 놓칩니다. 의미 검색은 임베딩이 법률 텍스트로 학습되지 않았다면 놓칩니다. 실제 검색은 두 가지 모두를 처리해야 합니다.

IR(정보 검색)은 모든 RAG 시스템, 검색 창, 문서 사이트의 퍼지 검색(fuzzy lookup) 아래에 있는 파이프라인입니다. 2026년에 프로덕션에서 작동하는 아키텍처는 단일 방법이 아닙니다. 이는 상호 보완적인 방법들의 체인으로, 각각 이전 방법의 실패 사례를 잡아냅니다.

이 강의에서는 각 구성 요소를 구축하고 각 방법이 어떤 실패 사례를 처리하는지 설명합니다.

## 개념

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

4개의 계층. 필요한 것을 선택하세요.

1. **희소 검색(BM25).** 빠르고 정확한 일치에 강점이 있지만 의미론적 유사성은 약합니다. 역색인(inverted index) 기반으로 실행됩니다. 수백만 개의 문서에서 쿼리당 10ms 미만의 응답 속도를 제공합니다. 법령 참조, 제품 코드, 오류 메시지, 명명된 엔티티 등을 정확히 찾아줍니다.
2. **밀집 검색(dense retrieval).** 쿼리와 문서를 벡터로 인코딩합니다. 최근접 이웃 검색(nearest neighbor search)을 수행합니다. 동의어 및 의미적 유사성을 포착하지만, 한 글자 차이로 다른 정확한 키워드 일치는 놓칩니다. FAISS 또는 벡터 DB를 사용할 경우 쿼리당 50-200ms가 소요됩니다.
3. **퓨전(fusion).** 희소 및 밀집 검색의 순위 목록을 통합합니다. 상호 순위 융합(Reciprocal Rank Fusion, RRF)은 원시 점수(서로 다른 스케일로 존재)를 무시하고 순위 위치만 사용하기 때문에 쉬운 기본 옵션입니다. 가중치 융합(weighted fusion)은 특정 도메인에 한 신호가 우세할 때 선택할 수 있습니다.
4. **크로스-인코더 재순위화(cross-encoder rerank).** 퓨전에서 상위 30개 결과를 가져옵니다. 크로스-인코더(쿼리 + 문서를 함께 입력으로 받아 각 쌍에 점수 부여)를 실행합니다. 상위 5개를 유지합니다. 크로스-인코더는 쌍당 처리 속도가 바이-인코더(bi-encoder)보다 느리지만 훨씬 정확합니다. 상위 30개 결과에만 실행함으로써 비용을 분산시킵니다.

3-way 검색(BM25 + 밀집 검색 + SPLADE 같은 학습 기반 희소 검색)은 2026 벤치마크에서 2-way보다 우수하지만, 학습 기반 희소 인덱스를 위한 인프라가 필요합니다. 대부분의 팀에게는 2-way에 크로스-인코더 재순위화를 추가하는 것이 최적의 조합입니다.

## 구축 방법

### 1단계: BM25 직접 구현

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

알아둘 만한 두 가지 파라미터. `k1=1.5`는 용어 빈도 포화를 제어하며, 값이 높을수록 용어 반복에 더 많은 가중치가 부여됩니다. `b=0.75`는 길이 정규화를 제어하며, 0은 문서 길이를 무시하고 1은 완전히 정규화합니다. 기본값은 원 논문의 Robertson 권장값이며 거의 조정이 필요 없습니다.

### 2단계: 바이-인코더를 사용한 밀집 검색

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2-정규화된 임베딩은 내적(dot product)이 코사인 유사도와 같도록 합니다. `all-MiniLM-L6-v2`는 384차원이며 빠르고 대부분의 영어 검색에 충분히 강력합니다. 다국어 작업에는 `paraphrase-multilingual-MiniLM-L12-v2`를 사용하세요. 최고 정확도를 원한다면 `bge-large-en-v1.5` 또는 `e5-large-v2`를 사용하세요.

### 3단계: 상호 순위 융합(Reciprocal Rank Fusion)

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 상수는 원본 RRF 논문에서 유래했습니다. `k`가 높을수록 순위 차이의 기여도가 평평해지고, 낮을수록 상위 순위가 지배적입니다. 60은 출판된 기본값이며 거의 조정이 필요 없습니다.

### 4단계: 하이브리드 검색 + 재순위

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

세 단계가 결합되었습니다. BM25는 어휘적 일치를 찾습니다. 밀집 검색은 의미적 일치를 찾습니다. RRF는 점수 보정 없이도 두 순위를 병합합니다. 크로스-인코더는 상위 30개를 쿼리-문서 쌍으로 재점수화하며, 이는 바이-인코더가 놓친 세밀한 관련성을 포착합니다. 상위 5개를 유지합니다.

### 5단계: 평가

| 메트릭 | 의미 |
|--------|---------|
| Recall@k | 정답 문서가 존재하는 쿼리 중, 상위 k개에 포함되는 비율 |
| MRR (Mean Reciprocal Rank) | 첫 번째 관련 문서의 1/순위 평균 |
| nDCG@k | 이진 관련성(관련/비관련)이 아닌 관련성 등급을 고려 |

RAG의 경우, 검색기의 **Recall@k**가 가장 중요한 수치입니다. 검색된 집합에 올바른 구절이 없으면 리더(reader)가 답변할 수 없습니다.

디버깅 팁: 실패한 쿼리에 대해 희소 검색과 밀집 검색 순위를 비교하세요. 한 방법은 정답 문서를 찾고 다른 방법은 찾지 못한다면, 어휘 불일치(해결: 누락된 반쪽 추가) 또는 의미적 모호성(해결: 더 나은 임베딩 또는 재순위기) 문제가 있을 수 있습니다.

## 사용 방법

2026년 스택:

| 규모 | 스택 |
|-------|-------|
| 1k-100k 문서 | 인메모리 BM25 + `all-MiniLM-L6-v2` 임베딩 + RRF. 별도 DB 없음. |
| 100k-10M 문서 | FAISS 또는 pgvector(밀집) + Elasticsearch / OpenSearch(BM25). 병렬 실행. |
| 10M+ 문서 | Qdrant / Weaviate / Vespa / Milvus(하이브리드 지원). 상위 30개 항목에 대해 크로스-인코더 재랭킹. |
| 최고 품질 프론티어 | 3-way(BM25 + 밀집 + SPLADE) + ColBERT 후기 상호작용 재랭킹 |

무엇을 선택하든 평가 예산을 확보하라. 엔드투엔드 RAG 정확도 벤치마킹 전에 검색 재현율을 벤치마크하라. 검색기가 놓친 것은 리더기가 고칠 수 없다.

### 2026년 프로덕션 RAG에서 얻은 교훈

- **RAG 실패의 80%는 모델이 아닌 수집(ingestion)과 청킹(chunking)에서 비롯된다.** 팀은 LLM을 교체하고 프롬프트를 튜닝하는 데 몇 주를 소비하지만, 검색기는 조용히 세 번째 쿼리마다 잘못된 컨텍스트를 반환한다. 청킹을 먼저 수정하라.
- **청킹 전략은 청크 크기보다 중요하다.** 고정 크기 분할은 표, 코드, 중첩 헤더를 망가뜨린다. 문장 인식(sentence-aware)이 기본값이며, 기술 문서와 제품 매뉴얼에는 의미론적 또는 LLM 기반 청킹이 효과적이다.
- **부모-문서 패턴.** 정밀도를 위해 작은 "자식" 청크를 검색하라. 동일한 부모 섹션의 여러 자식이 나타날 때, 부모 블록을 교체하여 컨텍스트를 보존하라. 이는 재학습 없이 답변 품질을 일관되게 향상시킨다.
- **k_rerank=3이 일반적으로 최적값이다.** 그 이상의 청크는 토큰 비용과 생성 지연만 증가시킬 뿐 답변 품질을 높이지 않는다. 만약 k=8이 여전히 k=3보다 낫다면 재랭커가 제대로 작동하지 않는 것이다.
- **HyDE / 쿼리 확장.** 쿼리로부터 가상의 답변을 생성하고, 이를 임베딩하여 검색하라. 짧은 질문과 긴 문서 사이의 표현 차이를 해소한다. 훈련 없이도 정밀도를 무료로 향상시킬 수 있다.
- **8K 토큰 미만의 컨텍스트 예산.** 해당 한계에서 지속적인 히트(hit)는 재랭커 임계값이 너무 느슨함을 의미한다.
- **모든 것을 버전 관리하라.** 프롬프트, 청킹 규칙, 임베딩 모델, 재랭커. 어떤 변동도 답변 품질을 조용히 망가뜨린다. 충실도, 컨텍스트 정밀도, 미해결 질문 비율에 대한 CI 게이트는 사용자가 문제를 인지하기 전에 회귀를 차단한다.
- **3-way 검색(BM25 + 밀집 + SPLADE 같은 학습형 희소)은 2-way보다 2026년 벤치마크에서 우수하다.** 특히 고유 명사와 의미를 혼합한 쿼리에 효과적이다. 인프라가 SPLADE 인덱스를 지원할 때 이를 도입하라.

적절한 검색 설계는 2026년 업계 측정에 따르면 환각(hallucination)을 70-90% 감소시킨다. 대부분의 RAG 성능 향상은 모델 파인튜닝이 아닌 더 나은 검색에서 비롯된다.

## Ship It

`outputs/skill-retrieval-picker.md`로 저장:

```markdown
---
name: retrieval-picker
description: 주어진 코퍼스와 쿼리 패턴에 대한 검색 스택을 선택합니다.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

요구사항(코퍼스 크기, 쿼리 패턴, 지연 시간 예산, 품질 기준, 인프라 제약)을 고려하여 다음을 출력합니다:

1. 스택. BM25 단독, 밀집 검색(dense) 단독, 하이브리드(BM25 + 밀집 + RRF), 하이브리드 + 교차 인코더 재순위, 또는 3-way(BM25 + 밀집 + 학습 기반 희소 검색).
2. 밀집 인코더. 특정 모델 이름을 지정합니다. 언어, 도메인, 컨텍스트 길이에 맞게 매칭합니다.
3. 재순위기. 사용하는 경우 특정 교차 인코더 모델 이름을 지정합니다. 재순위는 상위 30개 결과에 30-100ms 지연 시간을 추가함을 명시합니다.
4. 평가 계획. Recall@10이 주요 검색기 지표입니다. 다중 정답의 경우 MRR을 사용합니다. 먼저 기준선(baseline)을 설정한 후, 기준선 대비 점진적 개선 사항을 측정합니다.

명명된 엔티티, 오류 코드, 제품 SKU가 포함된 코퍼스에 대해 사용자가 밀집 검색이 정확한 매칭을 처리한다는 증거를 제시하지 않는 한, 밀집 검색 단독을 권장하지 않습니다. 최종 상위 5개 결과가 사용자 답변을 결정하는 고위험 검색(법률, 의료) 분야에서 재순위 단계를 생략하는 것을 거부합니다.
```

## 연습 문제

1. **쉬움.** 500개 문서 코퍼스에 대해 위의 `hybrid_search`를 구현하세요. 20개의 쿼리를 테스트하세요. BM25-only, dense-only, hybrid 간 5위까지의 재현율(recall)을 비교하세요.
2. **중간.** MRR(Mean Reciprocal Rank) 계산을 추가하세요. 각 테스트 쿼리에 대해 정답 문서의 순위를 BM25, dense, hybrid 순위에서 찾고, 각각에 대한 MRR을 보고하세요.
3. **어려움.** MultipleNegativesRankingLoss(Sentence Transformers)를 사용하여 도메인에 맞춰 dense 인코더를 파인튜닝(fine-tuning)하세요. 500개의 쿼리-문서 쌍으로 학습 세트를 구성하세요. 파인튜닝 전후의 재현율(recall)을 비교하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| BM25 | 키워드 검색 | 오카피 BM25(Okapi BM25). 용어 빈도(term frequency), 역문서 빈도(IDF), 문서 길이를 기반으로 문서 점수를 매김. |
| Dense retrieval | 벡터 검색 | 쿼리와 문서를 벡터로 인코딩한 후 가장 가까운 이웃을 찾음. |
| Bi-encoder | 임베딩 모델 | 쿼리와 문서를 독립적으로 인코딩. 쿼리 시 빠른 속도. |
| Cross-encoder | 재순위 모델(reranker model) | 쿼리와 문서를 함께 인코딩. 느리지만 정확함. |
| RRF | 순위 융합(rank fusion) | 두 순위를 `1/(k + rank)` 합계로 결합. |
| Recall@k | 검색 메트릭(retrieval metric) | 관련 문서가 상위 k개 안에 있는 쿼리의 비율.>

## 추가 자료

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25에 대한 권위 있는 논문.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — 표준 bi-encoder인 DPR.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 밀집 검색과의 격차를 줄인 학습 기반 희소 검색기.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 논문.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — 후기 상호작용 검색.