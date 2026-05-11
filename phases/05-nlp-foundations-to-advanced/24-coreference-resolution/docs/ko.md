# 코참조 해결(Coreference Resolution)

> "그녀가 그에게 전화를 걸었다. 그는 답하지 않았다. 의사는 점심 중이었다." 두 사람을 가리키는 세 개의 참조가 있지만 아무도 이름이 언급되지 않았다. 코참조 해결은 누가 누구인지 파악한다.

**유형:** 학습
**언어:** Python
**선수 지식:** Phase 5 · 06 (개체명 인식(NER)), Phase 5 · 07 (품사 태깅(POS) & 파싱)
**소요 시간:** ~60분

## 문제 정의

300단어 분량의 기사에서 Apple Inc.에 대한 모든 언급을 추출하는 작업. 기사에 "Apple"이라고 명시되어 있으면 쉽지만, "the company", "they", "Cupertino's technology giant", "Jobs's firm"과 같이 표현될 경우 어려워집니다. 이러한 언급을 동일한 개체로 연결하지 않으면 NER 파이프라인이 언급의 60-80%를 놓치게 됩니다.

공동참조 해결(Coreference Resolution)은 동일한 실제 개체를 가리키는 모든 표현을 하나의 클러스터로 연결합니다. 이는 표면 수준의 NLP(NER, 파싱)와 다운스트림 의미 분석(정보 추출, 질의응답, 요약, 지식 그래프) 사이의 접착제 역할을 합니다.

2026년에 중요한 이유:

- **요약**: "The CEO announced..." vs "Tim Cook announced..." — 요약문에는 CEO의 이름이 명시되어야 합니다.
- **질의응답**: "Who did she call?"과 같은 질문에는 "she"를 해결해야 합니다.
- **정보 추출**: "PER1 founded Apple"과 "Jobs founded Apple"을 별도의 항목으로 저장하는 지식 그래프는 잘못되었습니다.
- **다중 문서 정보 추출**: 동일한 이벤트에 대한 여러 기사 간 언급을 병합하려면 문서 간 공동참조 해결이 필요합니다.

## 개념

![Coreference clustering: mentions → entities](../assets/coref.svg)

**과제.** 입력: 문서. 출력: 각 클러스터가 하나의 엔티티를 참조하는 멘션(span)들의 클러스터링.

**멘션 유형.**

- **명명된 엔티티.** "Tim Cook"
- **명사구.** "the CEO", "the company"
- **대명사.** "he", "she", "they", "it"
- **동격.** "Tim Cook, Apple's CEO,"

**아키텍처.**

1. **규칙 기반 (Hobbs, 1978).** 구문 트리 기반 대명사 해결. 문법 규칙 사용. 좋은 베이스라인. 대명사에서 이기기 의외로 어려움.
2. **멘션 쌍 분류기.** 모든 멘션 쌍 (m_i, m_j)에 대해 코퍼퍼런스 여부 예측. 추이적 폐포로 클러스터링. 2016년 이전 표준.
3. **멘션 순위 지정.** 각 멘션에 대해 후보 선행사("선행사 없음" 포함) 순위 매기기. 최상위 선택.
4. **스팬 기반 엔드투엔드 (Lee et al., 2017).** 트랜스포머 인코더. 길이 제한까지 모든 후보 스팬 열거. 멘션 점수 예측. 각 스팬에 대한 선행사 확률 예측. 탐욕적 클러스터링. 현대적 기본 접근법.
5. **생성형 (2024+).** LLM에 프롬프트: "이 텍스트의 모든 대명사와 그 선행사를 나열하세요." 쉬운 케이스에서 잘 작동, 긴 문서와 희귀 참조에서는 어려움.

**평가 지표.** 단일 지표로 클러스터링 품질을 포착할 수 없어 5가지 표준 지표(MUC, B³, CEAF, BLANC, LEA) 사용. 처음 3개 평균을 CoNLL F1로 보고. 2026년 CoNLL-2012에서 최신 기술: ~83 F1.

**알려진 어려운 케이스.**

- 페이지 이전에 소개된 엔티티를 참조하는 한정 기술구.
- 브리징 아나포라("the wheels" → 이전에 언급된 자동차).
- 중국어 및 일본어와 같은 언어의 제로 아나포라.
- 카타포로(참조 대상보다 대명사가 먼저 나옴): "When **she** walked in, Mary smiled."

## 구축 방법

### 1단계: 사전 훈련된 신경 공동 참조 모델 (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # 실험 모델
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

더 긴 문서에서는 다음과 같은 결과를 얻을 수 있습니다:
- 클러스터 1: [Apple, The company, they]
- 클러스터 2: [new products]

### 2단계: 규칙 기반 대명사 해결기 (교육용)

`code/main.py`에서 표준 라이브러리만 사용하는 구현체를 확인하세요:

1. 언급 추출: 고유 명사(대문자로 시작하는 구간), 대명사(딕셔너리 조회), 한정 기술어("the X").
2. 각 대명사에 대해 이전 K개 언급을 살펴보고 다음 기준으로 점수화:
   - 성/수 일치(휴리스틱)
   - 최근성(가까운 것이 우선)
   - 통사적 역할(주어 선호)
3. 가장 높은 점수를 받은 선행사를 연결.

신경 모델과는 경쟁력이 없습니다. 하지만 검색 공간과 종단간 모델이 내려야 하는 결정들을 보여줍니다.

### 3단계: LLM을 이용한 공동 참조 해결

```python
prompt = f"""Text: {text}

사람이나 회사를 가리키는 모든 대명사와 명사구를 나열하세요.
그것들이 가리키는 대상별로 클러스터링하세요. JSON 형식으로 출력:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

주의해야 할 두 가지 실패 모드. 첫째, LLM이 과도하게 병합("him"과 "her"이 서로 다른 두 사람을 가리킬 때). 둘째, LLM이 긴 문서에서 언급을 조용히 생략. 항상 구간 오프셋 검사로 검증하세요.

### 4단계: 평가

표준 conll-2012 스크립트는 MUC, B³, CEAF-φ4를 계산하고 평균을 보고합니다. 내부 평가에서는 주석 처리된 테스트 세트에 대한 구간 수준 정밀도와 재현율로 시작한 후 언급 연결 F1을 추가하세요.

## 함정

- **싱글톤 폭발(Singleton explosion).** 일부 시스템은 모든 언급을 개별 클러스터로 보고합니다. B³는 관대한 반면, MUC는 이를 엄격히 평가합니다. 항상 세 가지 메트릭을 모두 확인하세요.
- **긴 컨텍스트의 대명사.** 2,000토큰 이상의 문서에서 성능이 ~15 F1 포인트 하락합니다. 청크를 신중하게 구성하세요.
- **성별 가정.** 하드코딩된 성별 규칙은 논바이너리 참조, 조직, 동물에 대해 실패합니다. 학습된 모델이나 중립적 점수 매기기를 사용하세요.
- **긴 문서에서의 LLM 드리프트.** 단일 API 호출로는 50개 이상의 단락에 걸친 언급을 안정적으로 클러스터링할 수 없습니다. 슬라이딩 윈도우 + 병합 방식을 사용하세요.

## 사용 방법

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 영어, 단일 문서 | `en_coreference_web_trf` (spaCy-experimental) 또는 AllenNLP 신경망 코레퍼럴 |
| 다국어 | OntoNotes 또는 Multilingual CoNLL로 학습된 SpanBERT / XLM-R |
| 문서 간 이벤트 코레퍼럴 | 특수 목적 엔드투엔드 모델 (2025–26 SOTA) |
| 빠른 LLM 기준선 | 구조화된 출력 코레퍼럴 프롬프트가 있는 GPT-4o / Claude |
| 프로덕션 대화 시스템 | 규칙 기반 폴백 + 신경망 주 시스템 + 중요 슬롯에 대한 수동 검토 |

2026년에 출시되는 통합 패턴: NER을 먼저 실행한 후 코레퍼럴을 실행하고, 코레퍼럴 클러스터를 NER 엔티티로 병합합니다. 다운스트림 작업에서는 언급당 하나의 엔티티가 아닌 클러스터당 하나의 엔티티를 확인합니다.

## Ship It

`outputs/skill-coref-picker.md`로 저장:

```markdown
---
name: coref-picker
description: 코참조 해결 접근법, 평가 계획, 통합 전략을 선택합니다.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, 정보 추출]
---

사용 사례(단일 문서/다중 문서, 도메인, 언어)가 주어졌을 때 다음을 출력합니다:

1. 접근법. 규칙 기반 / 신경망 스팬 기반 / LLM 프롬프트 기반 / 하이브리드. 한 문장 이유.
2. 모델. 신경망인 경우 체크포인트 이름.
3. 통합. 작업 순서: 토크나이징 → 개체명 인식(NER) → 코참조 해결 → 다운스트림 작업.
4. 평가. CoNLL F1(MUC + B³ + CEAF-φ4 평균)을 보유 세트에서 평가 + 20개 문서에 대한 수동 클러스터 검토.

2,000토큰 이상의 문서에 대해 슬라이딩 윈도우 병합 없이 LLM 전용 코참조 해결을 거부합니다. 언급 수준 정밀도-재현율 보고서 없이 코참조 해결을 실행하는 모든 파이프라인을 거부합니다. 인구통계학적으로 다양한 텍스트에 배포된 성별 휴리스틱 시스템을 플래그 처리합니다.
```

## 연습 문제

1. **쉬움.** `code/main.py`에 있는 규칙 기반 해결자(rule-based resolver)를 5개의 수작업 단락에 실행해 보세요. 정답(ground truth) 대비 언급-연결(mention-link) 정확도를 측정하세요.
2. **중간.** 뉴스 기사에 사전 학습된 신경망 코레퍼렌스 모델(pretrained neural coref model)을 적용해 보세요. 직접 수작업으로 주석(annotation)을 달아 클러스터(cluster)를 비교하세요. 어디서 실패했나요?
3. **어려움.** 코레퍼렌스 강화 NER 파이프라인(coref-enhanced NER pipeline)을 구축하세요: 먼저 NER을 실행한 후 코레퍼렌스 클러스터를 통해 병합(merge)합니다. 100개 기사에서 NER-온리(NER-only) 대비 엔티티 커버리지(entity-coverage) 개선도를 측정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| Mention | 참조 | 개체(이름, 대명사, 명사구)를 가리키는 텍스트 구간. |
| Antecedent | "it"이 가리키는 것 | 이후 언급이 공동 참조하는 이전 언급. |
| Cluster | 개체의 언급들 | 모두 동일한 실제 개체를 가리키는 언급들의 집합. |
| Anaphora | 역방향 참조 | 이후 언급이 이전 언급을 가리킴 ("he" → "John"). |
| Cataphora | 순방향 참조 | 이전 언급이 이후 언급을 가리킴 ("When he arrived, John..."). |
| Bridging | 암시적 참조 | "I bought a car. The wheels were bad." (그 차의 바퀴.) |
| CoNLL F1 | 리더보드의 숫자 | MUC, B³, CEAF-φ4 F1 점수의 평균.

## 추가 자료

- [Jurafsky & Martin, SLP3 Ch. 26 — 코참조 해결 및 엔티티 연결](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 표준 교과서 장.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — 스팬 기반 종단간(end-to-end) 접근.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 코참조 개선 사전 학습.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 벤치마크.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 규칙 기반 클래식.