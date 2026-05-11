# 다국어 NLP

> 하나의 모델, 100개 이상의 언어, 대부분의 언어에 대한 훈련 데이터 제로. 교차 언어 전이는 2020년대의 실용적인 기적입니다.

**유형:** 학습
**언어:** Python
**선수 지식:** Phase 5 · 04 (GloVe, FastText, 서브워드), Phase 5 · 11 (기계 번역)
**소요 시간:** ~45분

## 문제 정의

영어에는 수십억 개의 레이블된 예시가 있습니다. 우르두어는 수천 개, 마이틸리어(마이트일리)는 거의 없습니다. 글로벌 사용자를 대상으로 하는 실용적인 NLP 시스템은 작업별 훈련 데이터가 존재하지 않는 긴 꼬리(long tail) 언어들에서도 작동해야 합니다.

다국어 모델은 여러 언어를 동시에 하나의 모델로 훈련함으로써 이 문제를 해결합니다. 공유된 표현(shared representation)을 통해 모델은 고자원 언어에서 학습한 기술을 저자원 언어로 전이할 수 있습니다. 영어 감정 분석 작업에 모델을 파인튜닝(fine-tuning)하면, 우르두어에 대해서도 즉시 놀라운 수준의 감정 예측 결과를 생성합니다. 이것이 바로 제로샷(zero-shot) 교차 언어 전이(cross-lingual transfer)이며, 이는 NLP가 전 세계에 배포되는 방식을 재정의했습니다.

이 강의에서는 다국어 작업의 트레이드오프(tradeoff), 표준 모델(canonical models), 그리고 다국어 작업을 처음 시작하는 팀들을 자주 당황하게 만드는 한 가지 결정(전이용 소스 언어 선택)에 대해 설명합니다.

## 개념

![다국어 공유 임베딩 공간을 통한 교차 언어 전이](../assets/multilingual.svg)

**공유 어휘.** 다국어 모델은 모든 대상 언어의 텍스트로 학습된 SentencePiece 또는 WordPiece 토크나이저를 사용합니다. 어휘는 공유됩니다: 동일한 서브워드 단위는 관련 언어 간에 동일한 형태소를 나타냅니다. 영어와 이탈리아어의 `anti-`는 동일한 토큰을 얻습니다.

**공유 표현.** 여러 언어로 마스킹된 언어 모델링을 사전 학습한 트랜스포머는 의미적으로 유사한 다른 언어의 문장이 유사한 은닉 상태를 생성한다는 것을 학습합니다. mBERT, XLM-R, NLLB는 모두 이러한 특성을 보입니다. 영어의 "cat"에 대한 임베딩은 프랑스어의 "chat"과 스페인어의 "gato" 근처에 클러스터링되며, 전체 문장 임베딩도 마찬가지입니다.

**제로샷 전이.** 한 언어(일반적으로 영어)의 레이블된 데이터로 모델을 파인튜닝합니다. 추론 시 모델이 지원하는 다른 언어로 실행합니다. 대상 언어 레이블이 필요하지 않습니다. 결과는 언어학적으로 유사한 언어에서 강력하며 먼 언어에서는 약합니다.

**퓨샷 파인튜닝.** 대상 언어의 레이블된 예제 100-500개를 추가합니다. 분류 작업에서 정확도는 영어 기준치의 95-98%로 급상승합니다. 이는 다국어 NLP에서 가장 비용 효율적인 방법입니다.

## 모델

| 모델 | 연도 | 지원 언어 | 비고 |
|-------|------|----------|-------|
| mBERT | 2018 | 104개 언어 | 위키피디아로 학습. 최초의 실용적인 다국어 언어 모델. 저자원 언어에 약함. |
| XLM-R | 2019 | 100개 언어 | CommonCrawl로 학습 (위키피디아보다 훨씬 큼). 교차 언어 기준 설정. Base 2.7억, Large 5.5억 파라미터. |
| XLM-V | 2023 | 100개 언어 | 100만 토큰 어휘집 (vs 25만)을 가진 XLM-R. 저자원 언어에서 더 나은 성능. |
| mT5 | 2020 | 101개 언어 | 다국어 생성을 위한 T5 아키텍처. |
| NLLB-200 | 2022 | 200개 언어 | 메타의 번역 모델; 55개 저자원 언어 포함. |
| BLOOM | 2022 | 46개 언어 + 13개 프로그래밍 언어 | 다국어로 학습된 오픈 1760억 LLM. |
| Aya-23 | 2024 | 23개 언어 | 코히어의 다국어 LLM. 아랍어, 힌디어, 스와힐리어에서 강점. |

사용 사례에 따라 선택. 분류 작업은 XLM-R-base를 기본값으로 잘 작동. 생성 작업은 번역 시 NLLB, 오픈 생성 시 mT5를 사용. LLM 스타일 작업은 Aya-23 또는 명시적 다국어 프롬프팅을 사용하는 Claude와 조합.

## 소스 언어 결정 (2026 연구)

대부분의 팀은 파인튜닝 소스로 영어를 기본으로 선택합니다. 최근 연구(2026)에 따르면 이는 종종 잘못된 선택입니다.

언어 유사성은 원시 코퍼스 크기보다 전이 품질을 더 잘 예측합니다. 슬라브어군 대상 언어의 경우 독일어나 러시아어가 영어를 능가하는 경우가 많습니다. 인도어군 대상 언어의 경우 힌디어가 영어를 능가하는 경우가 많습니다. **qWALS** 유사성 지표(2026, World Atlas of Language Structures 기능 기반)는 이를 정량화합니다. **LANGRANK**(Lin et al., ACL 2019)는 언어 유사성, 코퍼스 크기, 계통적 관련성을 조합하여 후보 소스 언어를 순위 매기는 별도의 이전 방법입니다.

실용적 규칙: 대상 언어에 유형론적으로 가까운 고자원 언어가 있다면, 먼저 해당 언어로 파인튜닝을 시도한 후 영어 파인튜닝과 비교하세요.

## 구축 방법

### 1단계: 제로샷 다국어 분류

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

하나의 모델, 세 가지 언어, 동일한 API. NLI 데이터로 학습된 XLM-R은 함의 트릭을 통해 분류에 잘 전이됩니다.

### 2단계: 다국어 임베딩 공간

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

번역문들은 임베딩 공간에서 가까이 위치합니다. 다른 영어 문장은 더 멀리 위치합니다. 이것이 다국어 검색, 클러스터링, 유사도 작업을 가능하게 합니다.

### 3단계: 소수 샷 파인튜닝 전략

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

100-500개의 대상 언어 예시에 대해 `num_train_epochs=5`와 `learning_rate=2e-5`가 안전한 기본값입니다. 더 높은 학습률은 다국어 정렬을 붕괴시키고 영어 전용 모델을 생성하게 됩니다.

## 실제로 작동하는 평가 방법

- **보유한 언어별 정확도.** 집계하지 않음. 집계 시 긴 꼬리(long tail) 현상이 가려짐.
- **단일 언어 기준 모델과의 비교.** 충분한 데이터가 있는 언어의 경우, 처음부터 학습한 단일 언어 모델이 다국어 모델을 능가하기도 함. 테스트 필수.
- **엔티티 수준 테스트.** 대상 언어의 명명된 엔티티(named entity). 다국어 모델은 라틴 문자와 거리가 먼 스크립트(script)에 대해 약한 토크나이징(tokenization) 성능을 보이는 경우가 많음.
- **다국어 간 일관성.** 두 언어에서 동일한 의미는 동일한 예측을 생성해야 함. 성능 격차(gap)를 측정.

## 사용 방법

2026 스택:

| 작업 | 추천 모델 |
|-----|-----------|
| 분류, 100개 언어 | XLM-R-base (~270M) 파인튜닝 |
| 제로샷 텍스트 분류 | `joeddav/xlm-roberta-large-xnli` |
| 다국어 문장 임베딩 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 번역, 200개 언어 | `facebook/nllb-200-distilled-600M` (레슨 11 참조) |
| 생성형 다국어 | Claude, GPT-4, Aya-23, mT5-XXL |
| 저자원 언어 NLP | XLM-V 또는 관련 고자원 언어 도메인 특화 파인튜닝 |

성능이 중요한 경우 항상 대상 언어에서 파인튜닝을 위한 예산을 확보하세요. 제로샷은 시작점일 뿐, 최종 답이 아닙니다.

### 토크나이저 세금 (저자원 언어에서 발생하는 문제)

다국어 모델은 모든 언어에 대해 하나의 토크나이저를 공유합니다. 이 어휘집은 영어, 프랑스어, 스페인어, 중국어, 독일어로 주로 구성된 코퍼스로 학습됩니다. 주요 언어 집합 외부의 모든 언어에는 세 가지 세금이 조용히 누적됩니다:

- **비옥도 세금.** 저자원 언어 텍스트는 영어보다 단어당 훨씬 더 많은 토큰으로 분할됩니다. 힌디 문장 하나가 동등한 영어 문장보다 3-5배 더 많은 토큰을 필요로 할 수 있습니다. 이 3-5배 증가는 컨텍스트 윈도우, 학습 효율성, 지연 시간을 소모합니다.
- **변형 복구 세금.** 모든 오타, 발음 구별 기호 변형, 유니코드 정규화 불일치, 대소문자 변형은 임베딩 공간에서 전혀 관련 없는 시퀀스로 처리됩니다. 모델은 원어민이 당연히 아는 철자 대응 관계를 학습할 수 없습니다.
- **용량 초과 세금.** 세금 1과 2는 컨텍스트 위치, 레이어 깊이, 임베딩 차원을 소모합니다. 실제 추론을 위해 남는 자원은 동일한 모델에서 고자원 언어가 얻는 자원보다 체계적으로 더 작습니다.

실제 증상: 힌디로 모델을 정상적으로 학습시키고, 손실 곡선이 좋아 보이며, 평가 퍼플렉서티도 합리적일 수 있지만, 프로덕션 출력은 미묘하게 틀립니다. 문장 중간에 형태소가 붕괴됩니다. 희귀한 활용은 복구되지 않습니다. **고장난 토크나이저에서 데이터 확장으로 해결할 수 없습니다.**

완화 방안: 대상 언어에 대한 커버리지가 좋은 토크나이저 선택(XLM-V의 1M 토큰 어휘집은 직접적인 해결책); 학습 전 보류된 대상 텍스트의 토크나이저 비옥도 검증; 진정한 롱테일 스크립트의 경우 바이트 수준 폴백 사용(SentencePiece `byte_fallback=True`, GPT-2 스타일 바이트 수준 BPE)으로 OOV 방지.

## Ship It

`outputs/skill-multilingual-picker.md`로 저장:

```markdown
---
name: multilingual-picker
description: 다국어 NLP 작업을 위한 소스 언어, 대상 모델, 평가 계획 선택.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

주어진 요구사항(대상 언어, 작업 유형, 언어별 라벨 데이터 가용성)에 따라 다음을 출력:

1. 파인튜닝용 소스 언어. 기본값은 영어. 대상 언어와 언어학적으로 유사한 고자원 언어가 있는 경우 LANGRANK 또는 qWALS를 참고.
2. 베이스 모델. XLM-R(분류), mT5(생성), NLLB(번역), Aya-23(생성형 LLM).
3. Few-shot 예산. 가능한 경우 100-500개의 대상 언어 예시로 시작. 라벨링이 불가능한 경우에만 제로샷 사용.
4. 평가 계획. 언어별 정확도(집계 아님), 크로스링구얼 일관성, 비라틴 문자 기반 언어의 엔티티 수준 F1 점수.

언어별 평가 없이 다국어 모델을 출시하지 말 것 — 집계 메트릭은 롱테일 실패를 숨김. 토크나이저 커버리지가 낮은 문자(암하라어, 티그리냐어, 많은 아프리카 언어)는 바이트 폴백 지원 모델(SentencePiece의 byte_fallback=True 또는 GPT-2 같은 바이트 수준 토크나이저)이 필요함을 표시.
```

## 연습 문제

1. **쉬움.** 영어, 프랑스어, 힌디어, 아랍어 각각 10개 문장에 대해 제로샷 분류 파이프라인을 실행하세요. 각 언어별 정확도를 보고하세요. 프랑스어는 강력한 성능, 힌디어는 괜찮은 성능, 아랍어는 변동성이 있는 결과를 확인할 수 있을 것입니다.
2. **중간.** `paraphrase-multilingual-MiniLM-L12-v2`를 사용하여 소규모 다국어 코퍼스 위에 교차 언어 검색기를 구축하세요. 영어로 쿼리하고 모든 언어의 문서를 검색하세요. recall@5를 측정하세요.
3. **어려움.** 힌디어 분류 작업을 위해 영어 기반과 힌디어 기반 파인튜닝을 비교하세요. 두 방식 모두 500개의 대상 언어 예시를 사용하여 소량 파인튜닝을 수행하세요. 어떤 소스(영어/힌디어)가 더 나은 힌디어 정확도를 생성하는지, 그리고 그 차이가 얼마나 되는지 보고하세요. 이는 LANGRANK 논문을 축소한 실험입니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 다국어 모델(Multilingual model) | 하나의 모델, 여러 언어 | 언어 간 공유 어휘 및 파라미터. |
| 교차 언어 전이(Cross-lingual transfer) | 한 언어로 훈련, 다른 언어로 실행 | 소스 언어에서 파인튜닝(fine-tuning), 타겟 언어 라벨 없이 평가. |
| 제로샷(Zero-shot) | 타겟 언어 라벨 없음 | 타겟 언어 파인튜닝 없이 전이. |
| 퓨샷(Few-shot) | 소량의 타겟 라벨 | 100-500개의 타겟 언어 예시를 파인튜닝에 사용. |
| 엠버트(mBERT) | 최초의 다국어 언어 모델(LM) | 위키피디아로 사전 훈련된 104개 언어 BERT. |
| XLM-R | 표준 교차 언어 베이스라인 | CommonCrawl로 사전 훈련된 100개 언어 RoBERTa. |
| NLLB | 메타의 200개 언어 기계 번역(MT) | "No Language Left Behind". 55개 저자원 언어 포함.

## 추가 자료

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 논문.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 크로스링구얼 전이 연구 분야를 시작한 분석 논문.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 논문.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Cohere의 다국어 LLM인 Aya 모델.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 소스 언어 논문.