# 품사 태깅(POS Tagging)과 구문 분석(Syntactic Parsing)

> 문법은 한동안 유행하지 않았습니다. 그런 다음 모든 LLM 파이프라인이 구조화된 추출을 검증해야 했고, 다시 돌아왔습니다.

**유형:** 구축(Build)
**언어:** Python
**사전 요구 사항:** Phase 5 · 01 (텍스트 처리), Phase 2 · 14 (나이브 베이즈)
**소요 시간:** ~45분

## 문제 정의

레슨 01에서는 표제어 추출(lemmatization)에 품사 태그가 필요하다고 언급했습니다. `running`이 동사라는 것을 알지 못하면 표제어 추출기는 이를 `run`으로 변환할 수 없습니다. `better`가 형용사라는 것을 알지 못하면 `good`으로 변환할 수 없습니다.

이 약속은 전체 하위 분야를 숨기고 있었습니다. 품사 태깅은 문법적 범주를 할당합니다. 구문 분석(syntactic parsing)은 문장의 트리 구조를 복원합니다: 어떤 단어가 어떤 단어를 수식하는지, 어떤 동사가 어떤 논항을 지배하는지. 고전 자연어처리는 이 두 가지를 개선하는 데 20년을 보냈습니다. 그런 다음 딥러닝은 사전 훈련된 트랜스포머 위에 토큰 분류 작업으로 이들을 통합했고, 연구 커뮤니티는 다음 단계로 넘어갔습니다.

하지만 응용 커뮤니티는 그렇지 않습니다. 모든 구조화된 추출 파이프라인은 여전히 내부적으로 품사 태그와 의존 관계 트리를 사용합니다. LLM이 생성한 JSON은 문법적 제약 조건에 대해 검증됩니다. 질의응답 시스템은 의존 관계 파싱을 사용하여 쿼리를 분해합니다. 기계 번역 품질 평가기는 파싱 트리의 정렬을 확인합니다.

알아둘 가치가 있습니다. 이 레슨에서는 태그 집합(tagsets), 베이스라인, 그리고 직접 구현을 중단하고 spaCy를 호출하는 지점을 소개합니다.

## 개념

![POS 태그 + 의존성 파싱 예시](./assets/pos-parse.svg)

**품사 태깅(POS tagging)**은 각 토큰에 문법 범주를 라벨링합니다. **펜 트리뱅크(Penn Treebank, PTB)** 태그셋은 영어 기본값입니다. 36개의 태그로 구성되어 있으며, 일반 독자가 까다롭게 느낄 수 있는 구분이 있습니다: `NN` 단수 명사, `NNS` 복수 명사, `NNP` 고유 명사 단수, `VBD` 동사 과거형, `VBZ` 동사 3인칭 단수 현재형 등. **유니버설 디펜던시(Universal Dependencies, UD)** 태그셋은 더 조밀하며(17개 태그) 언어에 구애받지 않습니다. 다국어 작업의 기본값이 되었습니다.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**구문 파싱(Syntactic parsing)**은 트리를 생성합니다. 두 가지 주요 스타일:

- **구성성 파싱(Constituency parsing).** 명사구, 동사구, 전치사구가 서로 중첩됩니다. 출력은 비터미널 범주(NP, VP, PP)의 트리이며 단어가 리프 노드입니다.
- **의존성 파싱(Dependency parsing).** 각 단어는 문법 관계로 라벨이 지정된 단일 헤드 단어를 가집니다. 출력은 모든 간선이 (헤드, 의존어, 관계) 트리플인 트리입니다.

의존성 파싱은 2010년대에 승리했습니다. 특히 자유 어순 언어를 포함해 언어 간 깔끔하게 일반화하기 때문입니다.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 구축

### 1단계: 가장 빈도 높은 태그 기준선

동작하는 가장 단순한 품사 태거. 각 단어에 대해 훈련 데이터에서 가장 자주 등장했던 태그를 예측합니다.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Brown 코퍼스에서 이 기준선은 약 85% 정확도를 기록합니다. 좋지 않지만, 어떤 진지한 모델도 이 아래로 떨어져서는 안 되는 바닥선입니다.

### 2단계: 바이그램 HMM 태거

시퀀스의 결합 확률을 모델링합니다:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

두 개의 테이블: 전이 확률(이전 태그 주어진 현재 태그), 방출 확률(태그 주어진 단어). 라플라스 스무딩으로 카운트에서 둘 다 추정합니다. 비터비(태그 격자 위 동적 프로그래밍)로 디코딩합니다.

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown 코퍼스에서 바이그램 HMM은 약 93% 정확도를 기록합니다. 85%에서 93%로의 도약은 대부분 전이 확률 덕분입니다. 모델은 `DET NOUN`이 흔하고 `NOUN DET`은 드물다는 것을 학습합니다.

### 3단계: 현대 태거가 이를 능가하는 이유

전이 + 방출 확률은 지역적입니다. "I bought a saw"에서 `saw`는 명사지만 "I saw the movie"에서는 동사라는 것을 포착할 수 없습니다. 임의의 특징(접미사, 단어 형태, 앞뒤 단어, 단어 자체)을 가진 CRF는 약 97% 정확도를 기록합니다. BiLSTM-CRF나 트랜스포머는 98%+ 정확도를 기록합니다.

이 작업의 상한선은 주석자 간 불일치에 의해 결정됩니다. 인간 주석자는 Penn Treebank에서 약 97% 일치합니다. 98%를 넘는 모델은 아마도 테스트 세트에 과적합(overfitting)되고 있을 것입니다.

### 4단계: 의존 구문 분석 개요

처음부터 전체 의존 구문 분석을 구축하는 것은 범위를 벗어납니다. 교과서적 접근은 Jurafsky와 Martin에 있습니다. 알아야 할 두 가지 고전적 접근법:

- **전이 기반(transition-based)** 파서(arc-eager, arc-standard)는 시프트-리듀스 파서처럼 동작합니다: 토큰을 읽고 스택에 푸시한 후 리듀스 액션을 적용하여 아크를 생성합니다. 탐욕적 디코딩은 빠릅니다. 클래식 구현체는 MaltParser입니다. 현대 신경망 버전: Chen과 Manning의 전이 기반 파서.
- **그래프 기반(graph-based)** 파서(Eisner의 알고리즘, Dozat-Manning biaffine)는 가능한 모든 헤드-의존성 엣지에 점수를 매기고 최대 스패닝 트리를 선택합니다. 느리지만 더 정확합니다.

대부분의 응용 작업에는 spaCy를 호출하세요:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

`dep` 열을 아래에서 위로 읽으면 문장의 문법적 구조가 드러납니다.

## 사용 방법

모든 프로덕션 NLP 라이브러리는 표준 파이프라인의 일부로 품사 태거(POS tagger)와 의존 구문 분석기(dependency parser)를 제공합니다.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). 빠르고 정확하며 토크나이저 + 개체명 인식(NER) + 표제어 추출(lemmatization)과 통합됨. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (의존 관계).
- **Stanford NLP (stanza)**. CoreNLP의 후속 버전. 60개 이상의 언어에서 최신 기술 수준(state-of-the-art) 제공.
- **trankit**. 트랜스포머 기반, 우수한 UD 정확도.
- **NLTK**. `pos_tag`. 사용 가능하지만 느리고 오래된 기술. 교육용으로는 적합.

### 2026년에도 여전히 중요한 이유

- **표제어 추출(lemmatization).** 레슨 01에서는 정확한 표제어 추출을 위해 품사 태깅(POS)이 항상 필요합니다.
- **LLM 출력의 구조화된 추출.** 생성된 문장이 문법적 제약 조건(예: 주어-동사 일치, 필수 수식어)을 준수하는지 검증.
- **측면 기반 감성 분석.** 의존 구문 분석을 통해 어떤 형용사가 어떤 명사를 수식하는지 파악 가능.
- **쿼리 이해.** "웨스 앤더슨이 감독하고 빌 머레이가 출연한 영화"는 구문 분석을 통해 구조화된 제약 조건으로 분해됩니다.
- **다국어 전이 학습.** UD 태그와 의존 관계는 언어에 독립적이므로 새로운 언어의 제로샷 구조화된 분석이 가능.
- **저사양 파이프라인.** 트랜스포머를 사용할 수 없는 경우, 품사 태깅 + 의존 구문 분석 + 가제티어(gazetteer)만으로도 놀라운 결과를 얻을 수 있습니다.

## Ship It

`outputs/skill-grammar-pipeline.md`로 저장:

```markdown
---
name: grammar-pipeline
description: 다운스트림 NLP 작업을 위한 전통적인 품사(POS) + 의존 관계 파이프라인을 설계하세요.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

다운스트림 작업(정보 추출, 재구성 검증, 쿼리 분해, 표제어 추출)이 주어졌을 때 다음을 출력하세요:

1. 사용할 태그셋. 영어 전용 레거시 파이프라인에는 Penn Treebank, 다국어 또는 교차 언어 작업에는 Universal Dependencies를 사용합니다.
2. 라이브러리. 대부분의 프로덕션에는 spaCy, 학술용 다국어에는 stanza, 최고 UD 정확도에는 trankit을 사용합니다. 구체적인 모델 ID를 명시하세요.
3. 통합 패턴. 라이브러리를 호출하고 필요한 속성(`.pos_`, `.dep_`, `.head`)을 소비하는 3-5줄의 코드를 보여주세요.
4. 테스트할 실패 모드. 명사-동사 중의성(`saw`, `book`, `can`)과 전치사구(PP) 연결 중의성이 전형적인 함정입니다. 20개의 출력 샘플을 추출하고 직접 확인하세요.

자체 파서 개발을 권장하지 않습니다. 파서를 처음부터 구축하는 것은 연구 프로젝트이지 애플리케이션 작업이 아닙니다. 대소문자 변형을 처리하지 않고 POS 태그를 소비하는 모든 파이프라인을 취약하다고 표시하세요.
```

## 연습 문제

1. **쉬움.** 작은 태깅 코퍼스(예: NLTK의 Brown 서브셋)에서 가장 빈번한 태그 기준선(most-frequent-tag baseline)을 사용하여, 홀드아웃 문장에 대한 정확도를 측정하세요. 약 85%의 결과를 확인하세요.
2. **중간.** 위의 바이그램 HMM을 학습시키고 태그별 정밀도/재현율을 보고하세요. HMM이 가장 많이 혼동하는 태그는 무엇인가요?
3. **어려움.** spaCy의 의존성 구문 분석(dependency parse)을 사용하여 1000문장 샘플에서 주어-동사-목적어 삼중항(subject-verb-object triples)을 추출하세요. 수동으로 레이블링된 50개 삼중항으로 평가하세요. 추출이 실패하는 경우(수동태, 병렬구조, 생략된 주어 등)를 문서화하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| POS 태그 | 단어의 유형 | 문법적 범주. PTB는 36개, UD는 17개 태그 보유. |
| Penn Treebank | 표준 태그셋 | 영어 전용. 세부적인 동사 시제와 명사 수 구분. |
| Universal Dependencies | 다국어 태그셋 | PTB보다 더 포괄적; 언어 중립적; 다국어 작업 시 기본값. |
| Dependency parse | 문장 트리 | 각 단어는 하나의 헤드(head)를 가지며, 각 엣지(edge)는 문법적 관계를 나타냄. |
| Viterbi | 동적 프로그래밍 | 방출(emission) 확률과 전이(transition) 확률이 주어졌을 때, 가장 높은 확률의 태그 시퀀스를 찾음.

## 추가 학습 자료

- [Jurafsky and Martin — Speech and Language Processing, 8장과 18장](https://web.stanford.edu/~jurafsky/slp3/) — 품사 태깅(POS)과 구문 분석(parsing)에 대한 표준 교과서 설명.
- [Universal Dependencies 프로젝트](https://universaldependencies.org/) — 모든 다국어 파서에서 사용되는 언어 간 태깅 체계와 트리뱅크(treebank) 모음.
- [spaCy 언어학적 기능 가이드](https://spacy.io/usage/linguistic-features) — `Token` 객체에서 노출되는 모든 속성에 대한 실용적 참고 자료.
- [Chen and Manning (2014). 신경망을 활용한 빠르고 정확한 의존 구문 분석기](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 신경망 기반 파서를 주류로 만든 논문.