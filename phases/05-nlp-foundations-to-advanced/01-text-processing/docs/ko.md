# 텍스트 처리 — 토큰화, 표제어 추출, 표제어화

> 언어는 연속적이다. 모델은 이산적이다. 전처리는 그 사이의 다리다.

**유형:** 구축
**언어:** Python
**선수 지식:** Phase 2 · 14 (나이브 베이즈)
**소요 시간:** ~45분

## 문제 정의

모델은 "The cats were running."을 읽을 수 없습니다. 정수만 읽을 수 있습니다.

모든 NLP 시스템은 동일한 세 가지 질문으로 시작합니다. 단어의 시작은 어디인가. 단어의 어근은 무엇인가. "run", "running", "ran"을 도움이 될 때는 같은 것으로, 그렇지 않을 때는 다른 것으로 어떻게 처리할까.

토크나이제를 잘못 구현하면 모델은 쓰레기 데이터에서 학습하게 됩니다. 토크나이제가 `don't`를 하나의 토큰으로 처리하지만 `do n't`를 두 개의 토큰으로 처리하면 학습 분포가 분리됩니다. 스테머가 `organization`과 `organ`을 같은 어근으로 축소하면 토픽 모델링이 실패합니다. 표제어 추출기가 품사 맥락이 필요하지만 이를 전달하지 않으면 동사가 명사로 처리됩니다.

이 강의에서는 세 가지 전처리 기본 요소를 처음부터 구축한 다음, NLTK와 spaCy가 동일한 작업을 어떻게 수행하는지 보여줌으로써 트레이드오프를 이해할 수 있도록 합니다.

## 개념

세 가지 연산. 각각 고유한 역할과 실패 모드가 있습니다.

![전처리 파이프라인: 원시 텍스트 → 토큰 → 어간 또는 표제어 → 모델](./assets/pipeline.svg)

**토큰화(Tokenization)**는 문자열을 토큰으로 분할합니다. "토큰"은 의도적으로 모호하게 정의되는데, 적절한 세분성은 작업에 따라 달라지기 때문입니다. 고전 NLP에서는 단어 수준, 트랜스포머에서는 서브워드, 공백 없는 언어의 경우 문자 수준으로 처리합니다.

**어간 추출(Stemming)**은 규칙 기반으로 접미사를 잘라냅니다. 빠르고 공격적이지만 단순합니다. `running → run`. `organization → organ`. 두 번째 예시가 실패 모드입니다.

**표제어 추출(Lemmatization)**은 문법 지식을 활용해 단어를 사전 형태로 환원합니다. 느리지만 정확하며, 조회 테이블이나 형태소 분석기가 필요합니다. `ran → run` ("ran"이 "run"의 과거형임을 인지해야 함). `better → good` (비교급 형태를 인지해야 함).

경험적 규칙. 속도가 중요하고 노이즈를 허용할 수 있을 때(예: 검색 인덱싱, 대략적인 분류) 어간 추출을 사용합니다. 의미가 중요한 경우(예: 질의응답, 시맨틱 검색, 사용자가 읽을 모든 것) 표제어 추출을 사용합니다.

## 구축 방법

### 1단계: 정규 표현식 단어 토크나이저

가장 간단한 유용한 토크나이저는 알파벳과 숫자가 아닌 문자에서 분할하면서 구두점을 별도의 토큰으로 유지합니다. 완벽하지는 않지만 한 줄로 실행 가능한 기본 버전입니다.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

우선순위별 세 가지 패턴: 선택적 내부 아포스트로프가 있는 단어(`don't`, `it's`). 순수 숫자. 공백이나 알파벳/숫자가 아닌 단일 문자(구두점)를 독립 토큰으로 처리.

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

주목할 실패 사례. `3pm`은 `['3', 'pm']`으로 분할됩니다. 문자 시퀀스와 숫자 시퀀스가 번갈아 나타나기 때문입니다. 대부분의 작업에는 충분합니다. URL, 이메일, 해시태그는 모두 처리되지 않습니다. 프로덕션 환경에서는 일반 패턴 앞에 추가 패턴을 넣으세요.

### 2단계: 포터 스테머(1a 단계만)

전체 포터 알고리즘은 5단계의 규칙을 포함합니다. 1a 단계만으로도 가장 빈번한 영어 어미를 커버하며 패턴을 학습할 수 있습니다.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

규칙을 위에서 아래로 읽습니다. `ies -> i` 규칙 때문에 `ponies -> poni`가 됩니다. 실제 포터 알고리즘에는 이를 수정하는 1b 단계가 있습니다. 규칙 간 경쟁이 발생하며, 앞선 규칙이 우선합니다. 순서가 개별 규칙보다 더 중요합니다.

### 3단계: 조회 기반 표제어 추출기

표제어 추출은 형태론이 필요합니다. 교육용 버전은 작은 표제어 테이블과 폴백을 사용합니다.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

마지막 사례가 핵심 학습 포인트입니다. `watched`는 테이블에 없고 폴백은 `ing`만 처리합니다. 실제 표제어 추출은 `ed`, 불규칙 동사, 비교급 형용사, 발음 변화가 있는 복수형(`children -> child`)을 커버합니다. 프로덕션 시스템에서는 WordNet, spaCy의 형태론 분석기 또는 전체 형태소 분석기를 사용합니다.

### 4단계: 파이프라인 연결

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

누락된 부분은 품사 태거입니다. 5단계 · 07(품사 태깅)에서 구축합니다. 현재는 모든 품사를 `NOUN`으로 기본 설정하고 한계를 인정합니다.

## 사용 방법

NLTK와 spaCy는 프로덕션 버전을 제공합니다. 각각 몇 줄의 코드로 구현할 수 있습니다.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize`는 축약형, 유니코드, 정규식이 놓칠 수 있는 엣지 케이스를 처리합니다. `PorterStemmer`는 5단계 모두를 실행합니다. `WordNetLemmatizer`는 NLTK의 펜 트리뱅크(Penn Treebank) 체계에서 WordNet의 약어 집합으로 POS 태그를 변환해야 합니다. 위의 변환 배선은 대부분의 튜토리얼에서 생략되는 부분입니다.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy는 `nlp(text)` 뒤에 전체 파이프라인을 숨깁니다. 토크나이징, POS 태깅, 표제어 추출이 모두 실행됩니다. 대규모 작업에서 NLTK보다 빠릅니다. 기본 정확도가 더 높습니다. 단점은 개별 구성 요소를 쉽게 교체할 수 없다는 것입니다.

### 어떤 것을 선택할까

| 상황 | 선택 |
|-----------|------|
| 교육, 연구, 구성 요소 교체 | NLTK |
| 프로덕션, 다국어, 속도 중요 | spaCy |
| 트랜스포머 파이프라인 (어차피 모델 토크나이저로 토큰화할 것) | `tokenizers` / `transformers` 사용 및 클래식 전처리 생략 |

### 아무도 경고하지 않는 두 가지 실패 모드

대부분의 튜토리얼은 알고리즘만 가르치고 끝냅니다. 실제 전처리 파이프라인에서 문제가 되는 두 가지가 있는데, 거의 다루어지지 않습니다.

**재현성 드리프트.** NLTK와 spaCy는 버전 간 토크나이징 및 표제어 추출 동작을 변경합니다. spaCy 2.x에서 `['do', "n't"]`를 생성했던 것이 3.x에서는 `["don't"]`를 생성할 수 있습니다. 모델은 한 분포로 학습되었지만 추론은 다른 분포에서 실행됩니다. 정확도가 조용히 저하되고 아무도 이유를 모릅니다. `requirements.txt`에 라이브러리 버전을 고정하세요. 20개의 샘플 문장에 대한 예상 토크나이징을 동결하는 전처리 회귀 테스트를 작성하고, 업그레이드 시마다 실행하세요.

**학습/추론 불일치.** 공격적인 전처리(소문자화, 불용어 제거, 표제어 추출)로 학습하고, 원시 사용자 입력에 배포하면 성능이 급락합니다. 이는 프로덕션 NLP에서 가장 흔한 실패 사례입니다. 학습 중 전처리를 수행했다면 추론 시에도 동일한 함수를 실행해야 합니다. 전처리를 노트북 셀이 아닌 모델 패키지 내부의 함수로 제공하세요. 서빙 팀이 다시 작성하지 않도록 합니다.

## Ship It

엔지니어가 세 권의 교과서를 읽지 않고도 전처리 전략을 선택할 수 있도록 도와주는 재사용 가능한 프롬프트입니다.

`outputs/prompt-preprocessing-advisor.md`로 저장:

```markdown
---
name: preprocessing-advisor
description: NLP 작업을 위한 토크나이저, 표제어 추출, 표제어화 설정을 추천합니다.
phase: 5
lesson: 01
---

고전적인 NLP 전처리에 대한 조언을 제공합니다. 작업 설명을 제공받으면 다음을 출력합니다:

1. 토크나이저 선택 (정규 표현식, NLTK word_tokenize, spaCy, 또는 트랜스포머 토크나이저). 선택 이유를 설명하세요.
2. 표제어 추출(stemming), 표제어화(lemmatization), 둘 다, 또는 둘 다 하지 않을지 여부. 선택 이유를 설명하세요.
3. 구체적인 라이브러리 호출. 함수 이름을 명시하세요. NLTK가 관련된 경우 품사 태그(POS-tag) 번역을 인용하세요.
4. 사용자가 테스트해야 할 하나의 실패 모드.

사용자에게 보이는 텍스트에 대한 표제어 추출은 권장하지 않습니다. 품사 태그(POS tags) 없이 표제어화를 권장하지 않습니다. 비영어 입력은 다른 파이프라인이 필요하다고 표시하세요.
```

## 연습 문제

1. **쉬움.** `tokenize`를 확장하여 URL을 단일 토큰으로 유지하도록 수정하세요. 테스트: `tokenize("Visit https://example.com today.")`는 하나의 URL 토큰을 생성해야 합니다.
2. **중간.** Porter 단계 1b를 구현하세요. 단어에 모음이 포함되어 있고 `ed` 또는 `ing`로 끝나는 경우 이를 제거하세요. 이중 자음 규칙(`hopping -> hop`, `hopp`가 아님)을 처리하세요.
3. **어려움.** WordNet을 조회 테이블로 사용하지만 WordNet에 항목이 없는 경우 Porter 스테머로 폴백하는 표제어 추출기(lemmatizer)를 구축하세요. 태그가 지정된 코퍼스에서 정확도를 측정하여 순수 WordNet과 순수 Porter와 비교하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 토큰(Token) | 단어(Word) | 모델이 처리하는 기본 단위. 단어, 부분 단어(subword), 문자, 바이트 등이 될 수 있음. |
| 어간(Stem) | 단어의 어근(Root) | 규칙 기반 접미사 제거 결과. 항상 실제 단어는 아님. |
| 표제어(Lemma) | 사전 형태(Dictionary form) | 사전에서 찾을 수 있는 형태. 정확한 계산을 위해 문법적 맥락이 필요함. |
| 품사 태그(POS tag) | 품사(Part of speech) | 명사(NOUN), 동사(VERB), 형용사(ADJ) 등의 범주. 정확한 표제어 추출에 필요함. |
| 형태론(Morphology) | 단어 형태 규칙(Word shape rules) | 시제, 수, 격에 따라 단어 형태가 변하는 방식. 표제어 추출은 이에 의존함. |

## 추가 자료

- [Porter, M. F. (1980). 접미사 제거 알고리즘](https://tartarus.org/martin/PorterStemmer/def.txt) — 원본 논문, 5페이지, 여전히 가장 명확한 설명.
- [spaCy 101 — 언어학적 기능](https://spacy.io/usage/linguistic-features) — 실제 파이프라인의 구성 방식.
- [NLTK 책, 3장](https://www.nltk.org/book/ch03.html) — 아직 생각하지 못한 토크나이저 경계 사례.