# 임베딩 모델 — 2026 딥 다이브

> Word2Vec은 단어당 벡터를 제공했습니다. 현대 임베딩 모델은 구간당 벡터를 제공하며, 다국어 지원, 희소(sparse), 밀집(dense), 다중 벡터 뷰를 갖추고 인덱스 크기에 맞게 조정됩니다. 잘못 선택하면 RAG가 잘못된 것을 검색합니다.

**유형:** 학습
**언어:** Python
**선수 지식:** 5단계 · 03 (Word2Vec), 5단계 · 14 (정보 검색)
**소요 시간:** ~60분

## 문제

RAG 시스템의 40% 경우 잘못된 구절을 검색합니다. 이 문제의 원인은 벡터 데이터베이스나 프롬프트가 아닌 경우가 대부분입니다. 바로 임베딩 모델입니다.

2026년에 임베딩을 선택하는 것은 다음 5가지 축을 고려해야 함을 의미합니다:

1. **밀집(Dense) vs 희소(Sparse) vs 멀티벡터(Multi-vector).** 구절당 하나의 벡터, 토큰당 하나의 벡터, 또는 희소 가중치가 적용된 단어 집합.
2. **언어 커버리지.** 단일 언어(영어) 모델은 영어 전용 작업에서 여전히 우수합니다. 다국어 모델은 코퍼스가 혼합된 경우 승리합니다.
3. **컨텍스트 길이.** 512토큰 vs 8,192 vs 32,768 — 그리고 실제 유효 용량은 광고된 최대치의 60-70%인 경우가 많습니다.
4. **차원 예산.** 3,072개의 부동 소수점(전체 정밀도) = 벡터당 12KB. 1억 개 벡터 시 저장 비용은 월 $1,300입니다. 마트료시카 절단(Matryoshka truncation)은 이를 4배 줄일 수 있습니다.
5. **오픈(Open) vs 호스팅(Hosted).** 오픈 가중치(Open-weight)는 스택과 데이터를 직접 제어할 수 있음을 의미합니다. 호스팅은 최신 기술을 사용하는 대신 제어권을 포기하는 것을 의미합니다.

이 레슨은 지난 분기에 유행했던 것이 아닌, 증거를 바탕으로 선택할 수 있도록 트레이드오프를 명확히 합니다.

## 개념

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**밀집 임베딩(Dense embeddings).** 하나의 벡터당 하나의 문단(일반적으로 384-3,072 차원). 코사인 유사도(Cosine similarity)는 의미적 근접성에 따라 문단을 순위 매깁니다. OpenAI `text-embedding-3-large`, BGE-M3 밀집 모드, Voyage-3. 기본 선택 사항.

**희소 임베딩(Sparse embeddings).** SPLADE 스타일. 트랜스포머가 모든 어휘 토큰에 대한 가중치를 예측한 후 대부분을 0으로 만듭니다. 결과는 |어휘| 크기의 희소 벡터입니다. 키워드 중심의 쿼리(BM25와 유사)에 강력하지만 학습된 용어 가중치를 포함합니다.

**멀티벡터(지연 상호작용).** ColBERTv2, Jina-ColBERT. 토큰당 하나의 벡터. MaxSim으로 점수화: 각 쿼리 토큰에 대해 가장 유사한 문서 토큰을 찾고 점수를 합산합니다. 저장 및 점수화 비용이 더 비싸지만 긴 쿼리와 도메인 특화 코퍼스에서 우수합니다.

**BGE-M3: 세 가지 모드 동시 지원.** 단일 모델이 밀집, 희소, 멀티벡터 표현을 동시에 출력합니다. 각각 독립적으로 쿼리 가능하며, 점수는 가중치 합산을 통해 결합됩니다. 하나의 체크포인트에서 유연성을 원할 때 2026년 기본 선택.

**마트료시카 표현 학습(Matryoshka Representation Learning).** 벡터의 첫 N 차원이 유용한 독립 임베딩을 형성하도록 훈련됩니다. 1,536차원 벡터를 256차원으로 잘라내면 약 1%의 정확도 손실로 6배 저장 공간 절약이 가능합니다. OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+에서 지원.

### MTEB 리더보드는 부분적인 이야기만 전달합니다

Massive Text Embedding Benchmark — 출시 시(2022년) 8개 작업 유형에 걸쳐 56개 과제, MTEB v2에서 100+ 과제로 확장. 2026년 초, Gemini Embedding 2가 검색 분야(67.71 MTEB-R)에서 1위. Cohere embed-v4는 일반 분야(65.2 MTEB)에서 선두. BGE-M3는 오픈소스 다국어 분야(63.0)에서 선두. 리더보드는 필요하지만 충분하지 않습니다 — 항상 도메인에서 벤치마크하세요.

### 3단계 패턴

| 사용 사례 | 패턴 |
|----------|---------|
| 빠른 1차 필터링 | 밀집 양방향 인코더(BGE-M3, text-3-small) |
| 재현율 향상 | 희소(SPLADE, BGE-M3 희소) + RRF 결합 |
| 상위 50개 정확도 향상 | 멀티벡터(ColBERTv2) 또는 교차 인코더 재순위 지정기 |

대부분의 프로덕션 스택은 세 가지를 모두 사용합니다.

## 구축 방법

### 1단계: 기준선 — Sentence-BERT를 사용한 밀집 임베딩

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True`는 내적(dot product)을 코사인 유사도로 만듭니다. 항상 이 값을 설정하세요.

### 2단계: 마트료시카 트렁케이션

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

트렁케이션 후 재정규화하세요. Nomic v1.5, OpenAI text-3, Voyage-4는 처음 몇 단계에서 손실 없이 훈련됩니다. 비-마트료시카 모델(원래 Sentence-BERT)은 트렁케이션 시 급격히 성능이 저하됩니다.

### 3단계: BGE-M3 다기능성

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

세 가지 인덱스, 한 번의 추론 호출. 점수 융합:

```python
dense_score = ... # dense_vecs에 대한 코사인 유사도
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

도메인에 맞게 가중치를 조정하세요.

### 4단계: 사용자 정의 작업에서 MTEB 평가

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

후보 모델을 *대표성 있는* 하위 집합에서 실행하세요. 리더보드 순위만 신뢰하지 마세요 — 도메인이 중요합니다.

### 5단계: 처음부터 직접 구현한 코사인 유사도

`code/main.py` 참조. 평균 해시 트릭 임베딩(stdlib-only). 트랜스포머 임베딩과 경쟁할 수준은 아니지만, 형태를 보여줍니다: 토크나이징 → 벡터화 → 정규화 → 내적 계산.

## 주의 사항

- **쿼리와 문서에 동일한 모델 사용.** 일부 모델(Voyage, Jina-ColBERT)은 비대칭 인코딩을 사용합니다 — 쿼리와 문서가 서로 다른 경로를 통과합니다. 항상 모델 카드를 확인하세요.
- **접두사 누락.** `bge-*` 모델은 쿼리에 `"Represent this sentence for searching relevant passages: "`가 앞에 추가되어야 합니다. 이를 잊으면 3-5점 정도의 재현율 차이가 발생할 수 있습니다.
- **과도한 Matryoshka 축소.** 1,536 → 256은 일반적으로 안전합니다. 1,536 → 64는 그렇지 않습니다. 평가 세트에서 검증하세요.
- **컨텍스트 잘림.** 대부분의 모델은 최대 길이를 초과하는 입력을 조용히 자릅니다. 긴 문서는 청킹(chunking)이 필요합니다(레슨 23 참조).
- **지연 시간 꼬리 무시.** MTEB 점수는 p99 지연 시간을 숨깁니다. 600M 모델이 335M 모델보다 2점 더 높은 점수를 얻을 수 있지만 쿼리당 비용이 3배 더 들 수 있습니다.

## 사용 방법

2026 스택:

| 상황 | 선택 |
|-----------|------|
| 영어 전용, 빠른 속도, API | `text-embedding-3-large` 또는 `voyage-3-large` |
| 오픈소스 가중치, 영어 | `BAAI/bge-large-en-v1.5` |
| 오픈소스 가중치, 다국어 | `BAAI/bge-m3` 또는 `Qwen3-Embedding-8B` |
| 긴 컨텍스트(32k+ 토큰) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| CPU 전용 배포 | Nomic Embed v2 (137M 파라미터, MoE) |
| 저장 공간 제약 | Matryoshka-truncated + int8 양자화 |
| 키워드 중심 쿼리 | SPLADE 희소 임베딩 추가, 밀집 임베딩과 RRF 결합 |

2026 패턴: BGE-M3 또는 text-3-large로 시작, MTEB로 도메인 평가, 도메인 특화 모델이 3점 이상 우세할 경우 교체.

## Ship It

저장 위치: `outputs/skill-embedding-picker.md`

```markdown
---
name: embedding-picker
description: 주어진 코퍼스와 배포 환경에 적합한 임베딩 모델, 차원, 검색 모드를 선택합니다.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, 임베딩, 검색]
---

코퍼스(크기, 언어, 도메인, 평균 길이), 배포 대상(클라우드/엣지/온프레미스), 지연 시간 예산, 저장 공간 예산을 고려하여 다음을 출력합니다:

1. 모델. 체크포인트 이름 또는 API. 한 문장으로 이유 설명.
2. 차원. 전체 차원 / 마트료시카-절단 / int8-양자화. 저장 공간 예산과 연관된 이유.
3. 모드. 밀집(dense) / 희소(sparse) / 다중 벡터(multi-vector) / 하이브리드(hybrid). 이유.
4. 모델 카드에서 요구하는 경우 쿼리 접두사/템플릿.
5. 평가 계획. 도메인과 관련된 MTEB 작업 + nDCG@10을 사용한 보류 도메인 평가.

도메인 검증 없이 마트료시카를 64차원 미만으로 절단하는 추천은 거부합니다. 10,000개 미만의 패시지 코퍼스에 대한 ColBERTv2 추천은 거부합니다(과도한 오버헤드). 8,000토큰 이상의 장문 코퍼스가 512토큰 윈도우를 가진 모델로 라우팅되는 경우 플래그를 표시합니다.
```

## 연습 문제

1. **쉬움.** `bge-small-en-v1.5`로 100개 문장을 전체 차원(384)에서 인코딩한 후, Matryoshka 128에서 다시 인코딩합니다. 10개 쿼리에서 MRR(Mean Reciprocal Rank) 감소를 측정합니다.  
2. **중간.** 500개 도메인 패시지에서 BGE-M3 dense, sparse, colbert를 비교합니다. recall@10에서 어떤 방식이 승리합니까? RRF(Robertson–Zuccon–Zhou) 퓨전이 단일 모드 최고 성능을 능가합니까?  
3. **어려움.** 상위 2개 도메인 작업에서 후보 모델 3개를 대상으로 MTEB 평가를 실행합니다. MTEB 점수, 100개 쿼리 배치에서의 p99 지연 시간, $/1M 쿼리 비용을 보고합니다. 파레토 최적 모델을 선택합니다.  

> **참고:**  
> - MRR(Mean Reciprocal Rank): 평균 역순위  
> - RRF(Robertson–Zuccon–Zhou Fusion): 재순위화 기법  
> - p99: 99번째 백분위 수 지연 시간  
> - 파레토 최적: 성능-비용 트레이드오프에서 균형 잡힌 최적점

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| Dense embedding | 벡터 | 텍스트 당 하나의 고정 크기 벡터. 순위 매기기에 코사인 유사도 사용. |
| Sparse embedding | 학습된 BM25 | 어휘 토큰 당 하나의 가중치; 대부분 0; 종단간 훈련. |
| Multi-vector | ColBERT 스타일 | 토큰 당 하나의 벡터; MaxSim 점수 매기기; 더 큰 인덱스는 더 나은 재현율. |
| Matryoshka | 러시아 인형 트릭 | 첫 N차원은 그 자체로 유효한 더 작은 임베딩. |
| MTEB | 벤치마크 | 대규모 텍스트 임베딩 벤치마크 — 출시 시 56개 작업, v2에서 100+개. |
| BEIR | 검색 벤치마크 | 18개의 제로샷 검색 작업; 종종 도메인 간 강건성으로 인용됨. |
| Asymmetric encoding | 쿼리 ≠ 문서 경로 | 모델이 쿼리와 문서에 서로 다른 투영을 사용. |

## 추가 자료

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — bi-encoder 논문.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 리더보드 논문.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 통합 3가지 모드 모델.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 차원 사다리 훈련 목표.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 프로덕션에서의 후기 상호작용.
- [MTEB 리더보드 on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 실시간 순위.