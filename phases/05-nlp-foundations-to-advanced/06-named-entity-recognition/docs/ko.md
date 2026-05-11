# 개체명 인식(Named Entity Recognition)

> 이름을 뽑아내라. 모호한 경계, 중첩된 개체, 도메인 전문 용어를 다루기 전까지는 쉽게 들린다.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (단어 임베딩)
**소요 시간:** ~75분

## 문제 정의

"Apple sued Google over its iPhone search deal in the US." 이 문장에는 다섯 개의 엔티티가 있습니다: Apple (ORG), Google (ORG), iPhone (PRODUCT), search deal (아마도), US (GPE). 우수한 NER 시스템은 이 모든 엔티티를 정확한 유형과 함께 추출합니다. 반면 성능이 낮은 시스템은 iPhone을 놓치거나, Apple을 과일로 혼동하고, "US"를 PERSON으로 잘못 분류할 수 있습니다.

NER은 모든 구조화된 추출 파이프라인의 핵심 기술입니다. 이력서 파싱, 규정 준수 로그 스캔, 의료 기록 익명화, 검색 쿼리 이해, 챗봇 응답의 근거 확보, 법적 계약서 추출 등 다양한 분야에서 활용됩니다. NER은 눈에 띄지 않지만 항상 의존하는 기술입니다.

이 강의에서는 고전적인 접근법(규칙 기반, HMM, CRF)부터 현대적인 방법(BiLSTM-CRF, 트랜스포머)까지 단계별로 설명합니다. 각 단계는 이전 방법의 한계를 해결하며 발전해 왔습니다. 이러한 패턴이 바로 이 강의의 핵심 교훈입니다.

## 개념

![NER 태깅: BIO 스키마 + CRF+BiLSTM 파이프라인](./assets/ner.svg)

**BIO 태깅**(또는 BILOU)은 개체 추출을 시퀀스 라벨링 문제로 변환합니다. 각 토큰을 `B-TYPE`(개체의 시작), `I-TYPE`(개체 내부), 또는 `O`(어떤 개체에도 속하지 않음)로 라벨링합니다.

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

다중 토큰 개체 연쇄: `New B-GPE`, `York I-GPE`, `City I-GPE`. BIO를 이해하는 모델은 임의의 스팬을 추출할 수 있습니다.

아키텍처 발전 과정:

- **규칙 기반.** 정규식 + 가제티어 검색. 알려진 개체에 대한 높은 정밀도, 새로운 개체에 대한 제로 커버리지.
- **HMM.** 은닉 마르코프 모델. 태그 주어진 토큰의 방출 확률, 태그 간 전이 확률. 비터비 디코딩. 라벨된 데이터로 학습.
- **CRF.** 조건부 랜덤 필드. HMM과 유사하지만 판별적이므로 임의의 특징(단어 형태, 대문자 여부, 인접 단어)을 혼합할 수 있습니다. 2026년 기준 저자원 배포 환경에서 여전히 고전적인 프로덕션 워크호스.
- **BiLSTM-CRF.** 수작업 특징 대신 신경망 특징. LSTM이 문장을 양방향으로 읽고, CRF 레이어가 일관된 태그 시퀀스를 강제합니다.
- **트랜스포머 기반.** BERT를 토큰 분류 헤드로 파인튜닝. 최고 정확도. 가장 많은 연산 필요.

## 구축 방법

### 1단계: BIO 태깅 헬퍼 함수

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 2단계: 수작업 특징 추출

전통적인(비신경망) 개체명 인식(NER)에서 특징은 핵심입니다. 유용한 특징들:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")`은 `xXxxxx`를 반환합니다. `word_shape("USA-2024")`은 `XXX-dddd`를 반환합니다. 대문자 패턴은 고유명사에 대한 강력한 신호입니다.

### 3단계: 간단한 규칙 기반 + 사전 기반 베이스라인

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

실제 프로덕션 환경의 사전은 위키피디아와 DBpedia에서 스크래핑한 수백만 개의 항목을 포함합니다. 커버리지는 좋지만, 동음이의어 처리(예: 회사명 "Apple" vs 과일 "Apple")는 형편없습니다. 그래서 통계적 모델이 승리했습니다.

### 4단계: CRF 단계(개요, 전체 구현 아님)

확률 이론 기반 없이 50줄로 CRF를 처음부터 구현하는 것은 유익하지 않습니다. 대신 `sklearn-crfsuite`를 사용하세요:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1`과 `c2`는 L1 및 L2 정규화입니다. `all_possible_transitions=True`는 모델이 불법 시퀀스(예: `O` 다음에 `I-ORG`)가 발생할 가능성이 낮다는 것을 학습하도록 합니다. 이는 CRF가 BIO 일관성을 강제하는 방식입니다.

### 5단계: BiLSTM-CRF가 추가하는 기능

특징은 학습됩니다. 입력: 토큰 임베딩(GloVe 또는 fastText). LSTM은 좌에서 우로, 우에서 좌로 읽습니다. 연결된 은닉 상태는 CRF 출력 레이어를 통과합니다. CRF는 여전히 태그 시퀀스 일관성을 강제하며, LSTM은 수작업 특징을 학습된 특징으로 대체합니다.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 레이어에는 `torchcrf.CRF`(pytorch-crf 설치)를 사용하세요. 수작업 CRF 대비 성능 향상은 측정 가능하지만, 수만 개 이상의 레이블이 지정된 문장이 없다면 기대보다 작을 수 있습니다.

## 사용 방법

spaCy는 프로덕션 등급의 NER(Named Entity Recognition)을 기본 제공합니다.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

`iPhone`이 `PRODUCT`가 아닌 `ORG`로 레이블링된 것에 주목하세요 — spaCy의 소형 모델은 제품 엔티티 커버리지가 약합니다. 대형 모델(`en_core_web_lg`)은 더 나은 성능을 보입니다. 트랜스포머 모델(`en_core_web_trf`)은 더욱 우수합니다.

BERT 기반 NER을 위한 Hugging Face:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"`은 연속된 B-X, I-X 토큰을 하나의 스팬으로 병합합니다. 이 옵션이 없으면 토큰 수준 레이블이 반환되며 직접 병합해야 합니다.

### LLM 기반 NER (2026년 옵션)

제로샷 및 퓨샷 LLM NER은 많은 도메인에서 미세 조정 모델과 경쟁력 있는 성능을 보이며, 레이블 데이터가 부족할 때는 극적으로 우수합니다.

- **제로샷 프롬프팅.** LLM에 엔티티 유형 목록과 예시 스키마를 제공합니다. JSON 출력을 요청합니다. 즉시 사용 가능; 새로운 도메인에서 정확도는 보통 수준입니다.
- **ZeroTuneBio 스타일 프롬프팅.** 작업을 후보 추출 → 의미 설명 → 판단 → 재확인으로 분해합니다. 다단계 프롬프트(원샷이 아님)는 생의학 NER에서 정확도를 크게 향상시킵니다. 동일한 패턴이 법률, 금융, 과학 도메인에도 적용됩니다.
- **RAG를 활용한 동적 프롬프팅.** 모든 추론 호출 시 작은 주석이 달린 시드 세트에서 가장 유사한 레이블 예시를 검색합니다. 퓨샷 프롬프트를 실시간으로 구축합니다. 2026년 벤치마크에서 이 방법은 정적 프롬프팅 대비 GPT-4 생의학 NER F1을 11-12% 향상시킵니다.
- **엔티티 유형별 분해.** 긴 문서의 경우 모든 엔티티 유형을 한 번에 추출하면 길이가 증가함에 따라 재현율이 감소합니다. 엔티티 유형별 추출 패스를 별도로 실행합니다. 추론 비용은 증가하지만 정확도는 크게 향상됩니다. 이는 임상 노트와 법률 계약서의 표준 패턴입니다.

2026년 기준 프로덕션 권장 사항: 훈련 데이터를 수집하기 전에 LLM 제로샷 기준선으로 시작하세요. 종종 F1이 충분히 좋아 미세 조정이 필요 없을 수 있습니다.

### 클래식 NER이 여전히 우세한 경우

LLM이 사용 가능하더라도 클래식 NER이 우세한 경우:

- 지연 시간 예산이 50ms 미만일 때.
- 수천 개의 레이블 예시가 있고 98%+ F1이 필요할 때.
- 도메인이 안정적이며 사전 훈련된 CRF 또는 BiLSTM이 잘 전이될 때.
- 규제 제약으로 온프레미스, 비생성 모델이 필요할 때.

### 실패하는 경우

- **도메인 이동.** CoNLL 훈련 NER을 법률 계약서에 적용하면 가제터보다 성능이 나쁩니다. 도메인에 맞게 미세 조정하세요.
- **중첩 엔티티.** "Bank of America Tower"는 동시에 ORG와 FACILITY입니다. 표준 BIO는 중첩 스팬을 표현할 수 없습니다. 중첩 NER(다중 패스 또는 스팬 기반 모델)이 필요합니다.
- **긴 엔티티.** "United States Federal Deposit Insurance Corporation." 토큰 수준 모델은 때때로 이를 분할합니다. `aggregation_strategy`를 사용하거나 후처리하세요.
- **희소 유형.** DRUG_BRAND, ADVERSE_EVENT, DOSE와 같은 의료 NER 레이블. 범용 모델은 전혀 알지 못합니다. Scispacy와 BioBERT가 시작점입니다.

## Ship It

`outputs/skill-ner-picker.md`로 저장:

```markdown
---
name: ner-picker
description: 주어진 추출 작업에 적합한 NER 접근 방식을 선택합니다.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

작업 설명(도메인, 라벨 집합, 언어, 지연 시간, 데이터 양)이 주어졌을 때 다음을 출력합니다:

1. 접근 방식. 규칙 기반 + gazetteer, CRF, BiLSTM-CRF, 또는 트랜스포머 파인튜닝.
2. 시작 모델. 이름 지정 (spaCy 모델 ID, Hugging Face 체크포인트 ID, 또는 "custom, trained from scratch").
3. 라벨링 전략. BIO, BILOU, 또는 span 기반. 한 문장으로 정당화.
4. 평가. `seqeval` 사용. 항상 엔티티 수준 F1 보고 (토큰 수준 아님).

500개 미만의 라벨된 예시에 대해 트랜스포머 파인튜닝을 권장하지 않습니다. 단, 사용자가 이미 사전 훈련된 도메인 모델을 보유한 경우는 예외입니다. 중첩 엔티티는 span 기반 또는 다중 패스 모델이 필요하다고 표시합니다. 사용자가 "production scale"을 언급하고 라벨이 CoNLL-2003과 변경되지 않은 경우 gazetteer 감사를 요구합니다.
```

## 연습 문제

1. **쉬움.** `bio_to_spans` (즉, `spans_to_bio`의 역함수)를 구현하고 10개 문장에 대해 왕복 일관성(round-trip consistency)을 검증하세요.
2. **중간.** CoNLL-2003 영어 NER 데이터셋에서 위의 `sklearn-crfsuite` CRF를 학습시키세요. `seqeval`을 사용하여 엔티티별 F1 점수를 보고하세요. 일반적인 결과: ~84 F1.
3. **어려움.** 도메인 특화 NER 데이터셋(의료, 법률, 또는 금융)에서 `distilbert-base-cased`를 파인튜닝(fine-tuning)하세요. spaCy 소형 모델과 비교하세요. 데이터 누수(data leakage) 검증 절차를 문서화하고, 놀라운 점을 기록하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| NER | 이름 추출 | 토큰 스팬에 유형(PERSON, ORG, GPE, DATE, ...) 레이블을 지정. |
| BIO | 태깅 체계 | `B-X`는 시작, `I-X`는 계속, `O`는 외부. |
| BILOU | 향상된 BIO | `L-X`(마지막), `U-X`(단위) 추가로 경계 명확화. |
| CRF | 구조화된 분류기 | 레이블 간 전이 모델링, 단순 방출(emission) 이상. 유효한 시퀀스 강제. |
| 중첩 NER | 중첩 엔티티 | 한 스팬이 그 하위 스팬과 다른 엔티티일 수 있음. BIO로는 표현 불가. |
| 엔티티 수준 F1 | 적절한 NER 평가 지표 | 예측된 스팬이 실제 스팬과 정확히 일치해야 함. 토큰 수준 F1은 정확도를 과대평가. |

## 추가 자료

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 논문. 표준 참조.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 표준 토큰 분류 패턴을 소개한 논문.
- [spaCy 언어학적 기능 — 개체명](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 및 `Span`의 모든 속성에 대한 실용적 참고 자료.
- [seqeval](https://github.com/chakki-works/seqeval) — 올바른 평가 지표 라이브러리. 항상 사용하세요.