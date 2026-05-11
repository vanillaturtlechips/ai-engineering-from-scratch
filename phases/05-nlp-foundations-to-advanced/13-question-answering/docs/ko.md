# 질문 답변 시스템

> 세 가지 시스템이 현대 QA를 형성했습니다. 추출형(Extractive)은 텍스트 스팬(span)을 찾았습니다. 검색 증강형(Retrieval-augmented)은 문서 기반으로 답변을 생성했습니다. 생성형(Generative)은 답변을 직접 생성했습니다. 모든 현대 AI 어시스턴트는 이 세 가지의 혼합입니다.

**유형:** Build  
**언어:** Python  
**선수 지식:** Phase 5 · 11 (기계 번역), Phase 5 · 10 (어텐션 메커니즘)  
**소요 시간:** ~75분

## 문제 정의

사용자가 "첫 번째 iPhone은 언제 출시되었나요?"라고 입력하면 "2007년 6월 29일"이라는 답변을 기대합니다. "Apple의 역사는 길고 다양합니다" 같은 관련 없는 답변이나 "2007"처럼 문장 없이 고립된 단어가 아닙니다. 직접적이고 근거 있는 정확한 답변이 필요합니다.

지난 10년간 QA(Question Answering) 분야를 주도한 세 가지 아키텍처가 있습니다.

- **추출형 QA(Extractive QA).** 질문과 답변을 포함하는 것으로 알려진 지문이 주어졌을 때, 지문 내에서 답변의 시작 및 종료 위치 인덱스를 찾습니다. SQuAD(Stanford Question Answering Dataset)가 대표적인 벤치마크입니다.
- **개방형 QA(Open-domain QA).** 지문이 주어지지 않습니다. 먼저 관련 지문을 검색한 후 답변을 추출하거나 생성합니다. 이는 오늘날 모든 RAG(Retrieval-Augmented Generation) 파이프라인의 기반입니다.
- **생성형/폐쇄형 QA(Generative / Closed-book QA).** 대규모 언어 모델이 파라미터 메모리에서 직접 답변을 생성합니다. 검색 과정이 없어 추론 속도가 가장 빠르지만 사실 정확도는 가장 낮습니다.

2026년의 트렌드는 하이브리드 방식입니다. 최적의 몇 가지 지문을 검색한 후 생성형 모델에 해당 지문을 기반으로 답변하도록 프롬프트를 제공하는 것입니다. 이것이 RAG이며, 레슨 14에서는 검색 부분을 심층적으로 다룹니다. 이 레슨에서는 QA 부분을 구축합니다.

## 개념

![QA 아키텍처: 추출형, 검색 증강형, 생성형](../assets/qa.svg)

**추출형(Extractive).** 질문과 지문을 함께 트랜스포머(BERT 계열)로 인코딩합니다. 답변의 시작 및 종료 토큰 위치를 예측하는 두 개의 헤드를 훈련시킵니다. 손실은 유효한 위치에 대한 교차 엔트로피(cross-entropy)입니다. 출력은 지문에서 추출한 구간(span)입니다. 환각(hallucination)을 절대 발생시키지 않으며(구조상), 지문이 답변할 수 없는 질문은 처리하지 못합니다(구조상).

**검색 증강형(Retrieval-augmented, RAG).** 두 단계로 구성됩니다. 첫 번째, 검색기(retriever)가 코퍼스에서 상위 `k`개 지문을 찾습니다. 두 번째, 리더(reader, 추출형 또는 생성형)가 해당 지문을 활용해 답변을 생성합니다. 검색기-리더 분리는 각각을 독립적으로 훈련 및 평가할 수 있게 합니다. 현대 RAG는 종종 이 사이에 재순위기(reranker)를 추가합니다.

**생성형(Generative).** 디코더 전용 LLM(GPT, Claude, Llama)이 학습된 가중치를 기반으로 답변을 생성합니다. 검색 단계가 없습니다. 일반 상식에 뛰어나지만, 희귀하거나 최신 사실에서는 심각한 오류를 발생시킵니다. 환각(hallucination) 비율은 사전 훈련 데이터에서 사실 빈도와 반비례 관계를 가집니다.

## 구축 방법

### 1단계: 사전 훈련된 모델을 사용한 추출적 QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2`는 SQuAD 2.0으로 훈련되었으며, 답변할 수 없는 질문을 포함합니다. 기본적으로 `question-answering` 파이프라인은 모델의 null 점수가 승리할 때도 가장 높은 점수의 스팬을 반환합니다. 즉, 자동으로 빈 답변을 반환하지 않습니다. 명시적인 "답변 없음" 동작을 얻으려면 파이프라인 호출에 `handle_impossible_answer=True`를 전달하세요. 그러면 null 점수가 모든 스팬 점수를 초과할 때만 빈 답변을 반환합니다. 어떤 경우든 `score` 필드를 항상 확인하세요.

### 2단계: 검색 증강 파이프라인 (개요)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

2단계 파이프라인. 밀집 검색기(Sentence-BERT)는 의미적 유사도를 기반으로 관련 구절을 찾습니다. 추출적 리더(RoBERTa-SQuAD)는 상위 구절에서 답변 스팬을 추출합니다. 소규모 코퍼스에서 작동합니다. 백만 문서 코퍼스의 경우 FAISS 또는 벡터 데이터베이스를 사용하세요.

### 3단계: RAG를 사용한 생성적 QA

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

프롬프트 패턴이 중요합니다. 모델이 컨텍스트에 근거하도록 명시적으로 지시하고 컨텍스트가 부족할 때 "I don't know"를 반환하도록 하면 순진한 프롬프팅에 비해 환각률이 40-60% 감소합니다. 더 정교한 패턴은 인용, 신뢰도 점수, 구조화된 추출을 추가합니다.

### 4단계: 실제 환경을 반영한 평가

SQuAD는 **정확 일치(EM)**와 **토큰 수준 F1**을 사용합니다. EM은 정규화(소문자, 구두점 제거, 관사 제거) 후 엄격한 일치입니다. 예측이 정확히 일치하거나 0점을 받습니다. F1은 예측과 참조 간 토큰 중첩을 기반으로 계산되며 부분 점수를 부여합니다. 둘 다 다른 표현을 과소평가합니다. 예를 들어 "June 29, 2007" vs "June 29th, 2007"은 일반적으로 0 EM을 받지만(중첩이 정규화를 깨뜨림) 중첩 토큰으로 인해 상당한 F1을 얻습니다.

프로덕션 QA의 경우:

- **답변 정확도** (LLM 또는 인간 평가. 메트릭은 의미적 동등성을 포착하지 못함).
- **인용 정확도.** 인용된 구절이 실제로 답변을 지원하는지. 생성된 인용과 검색된 구절 간 문자열 일치로 자동 확인 가능.
- **거부 보정.** 답변이 검색된 구절에 없을 때 시스템이 "I don't know"라고 정확히 응답하는지. 거짓 신뢰도율 측정.
- **검색 재현율.** 리더 평가 전, 검색기가 상위 `k`에 올바른 구절을 포함하는지 측정. 리더가 누락된 구절을 수정할 수 없음.

### RAGAS: 2026년 프로덕션 평가 프레임워크

`RAGAS`는 RAG 시스템을 위해 특별히 제작되었으며 2026년 기본 제공 평가 도구입니다. 골드 참조 없이도 4가지 차원을 평가합니다:

- **충실도.** 답변의 각 주장이 검색된 컨텍스트에서 비롯되었는지. NLI 기반 함의로 측정. 주요 환각 메트릭.
- **답변 관련성.** 답변이 질문을 다루는지. 답변에서 가상의 질문을 생성하고 실제 질문과 비교하여 측정.
- **컨텍스트 정밀도.** 검색된 청크 중 실제로 관련 있는 비율. 낮은 정밀도 = 프롬프트의 노이즈.
- **컨텍스트 재현율.** 검색된 집합이 필요한 모든 정보를 포함하는지. 낮은 재현율 = 리더가 성공할 수 없음.

참조 없는 평가를 통해 선별된 골드 답변 없이도 실시간 프로덕션 트래픽을 평가할 수 있습니다. 정확한 일치 메트릭이 쓸모없는 개방형 질문의 경우 LLM-평가자를 추가로 계층화하세요.

`pip install ragas`. 검색기 + 리더를 연결. 쿼리당 4개의 스칼라 값 획득. 회귀에 대한 경고 발생.

## 사용 방법

2026년 스택.

| 사용 사례 | 권장 사항 |
|---------|-------------|
| 주어진 문단에서 답변 스팬 찾기 | `deepset/roberta-base-squad2` |
| 고정된 코퍼스 내에서, 폐쇄형(closed-book) 접근 방식 불가능 | RAG: 밀집 검색기(dense retriever) + LLM 리더(reader) |
| 문서 저장소에서 실시간 처리 | 하이브리드(BM25 + 밀집) 검색기 + 재순위 결정기(reranker) (레슨 14) |
| 대화형 QA(추가 질문) | 대화 기록 + 매 턴마다 RAG를 활용한 LLM |
| 고도로 사실 중심, 규제된 도메인 | 권위 있는 코퍼스 기반 추출형(extractive) 방식; 생성형(generative) 단독 사용 금지 |

2026년에는 LLM 기반 RAG가 더 많은 사례를 처리하기 때문에 추출형 QA가 유행하지 않습니다. 하지만 법적 연구, 규제 준수, 감사 도구와 같이 문자 그대로의 인용이 필요한 맥락에서는 여전히 사용됩니다.

## Ship It

`outputs/skill-qa-architect.md`로 저장:

```markdown
---
name: qa-architect
description: QA 아키텍처, 검색 전략, 평가 계획 선택.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

주어진 요구사항(코퍼스 크기, 질문 유형, 사실성 제약, 지연 시간 예산)에 따라 다음을 출력:

1. 아키텍처. 추출형(Extractive), 추출형 리더기(RAG with extractive reader), 생성형 리더기(RAG with generative reader), 또는 폐쇄형 LLM(closed-book LLM). 한 문장으로 이유 설명.
2. 검색기(Retriever). 없음(None), BM25, 밀집형(dense, 인코더 이름 명시), 또는 혼합형(hybrid).
3. 리더기(Reader). SQuAD-튜닝 모델, 이름 명시된 LLM, 또는 "도메인 파인튜닝 DistilBERT."
4. 평가. 추출형 벤치마크는 EM + F1; 프로덕션은 답변 정확도 + 인용 정확도 + 거부 보정(refusal calibration). 측정 항목과 측정 방법 명시.

규제 또는 컴플라이언스 민감 질문에 대해 폐쇄형 LLM 답변을 거부. 검색-재현율 기준(retrieval-recall baseline) 없는 QA 시스템 거부(검색기가 올바른 구절을 표면화했는지 모르면 리더기 평가 불가). 다중 추론(multi-hop reasoning)이 필요한 질문은 HotpotQA-훈련 시스템과 같은 전문 다중 추론 검색기 필요 표시.
```

## 연습 문제

1. **쉬움.** 10개의 위키피디아 문서에 대해 위의 SQuAD 추출적 파이프라인을 설정하세요. 10개의 질문을 직접 작성하세요. 정답이 얼마나 자주 나오는지 측정하세요. 문서와 질문이 정확하다면 7-9개가 정답일 것으로 예상됩니다.
2. **중간.** 거부 분류기를 추가하세요. 상위 검색 점수가 임계값(예: 0.3 코사인) 미만일 때 리더 모델을 호출하는 대신 "모르겠습니다"를 반환하세요. 임계값을 검증 세트에서 조정하세요.
3. **어려움.** 선택한 10,000개 문서 코퍼스에 대해 RAG 파이프라인을 구축하세요. RRF 융합(레슨 14 참조)을 사용한 하이브리드 검색(BM25 + 밀집)을 구현하세요. 하이브리드 단계 유무에 따른 답변 정확도를 측정하세요. 어떤 유형의 질문에서 가장 큰 이점을 얻는지 문서화하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 추출적 QA(Extractive QA) | 답변 구간 찾기 | 주어진 지문 내에서 답변의 시작 및 종료 인덱스 예측. |
| 오픈도메인 QA(Open-domain QA) | 코퍼스 기반 QA | 주어진 지문 없음; 검색 후 답변해야 함. |
| RAG(Retrieval-Augmented Generation) | 검색 후 생성 | 검색-증강 생성. 검색기(Retriever) + 판독기(Reader) 파이프라인. |
| SQuAD(Stanford Question Answering Dataset) | 표준 벤치마크 | 스탠포드 질문 답변 데이터셋. EM(Exact Match) + F1 메트릭. |
| 환각(Hallucination) | 지어낸 답변 | 검색된 문맥에 의해 뒷받침되지 않는 판독기 출력. |
| 거부 보정(Refusal calibration) | 침묵할 때 알기 | 시스템이 답변할 수 없을 때 "모릅니다"라고 정확히 말함. |

## 추가 자료

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — 벤치마크 논문.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — QA 분야의 표준 밀집 검색기(DPR).
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — RAG 개념을 명명한 논문.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 포괄적인 RAG 조사 논문.