# 엔티티 연결(Entity Linking) & 의미 명확화(Disambiguation)

> NER이 "Paris"를 발견했습니다. 엔티티 연결은 다음과 같이 결정합니다: 프랑스의 파리? 패리스 힐튼? 텍사스의 파리? 트로이의 왕자 파리? 연결이 없으면 지식 그래프는 모호한 상태로 남습니다.

**유형:** 구축(Build)
**언어:** Python
**선수 지식:** Phase 5 · 06 (NER), Phase 5 · 24 (공동 참조 해결)
**소요 시간:** ~60분

## 문제 정의

문장 "Jordan beat the press."에서 NER은 "Jordan"을 PERSON으로 태깅합니다. 좋은 결과입니다. 하지만 *어떤* Jordan일까요?

- 마이클 조던(농구 선수)?
- 마이클 B. 조던(배우)?
- 마이클 I. 조던(버클리 ML 교수 — 실제로 ML 논문에서 이런 혼란이 발생합니다)?
- 요르단(국가)?
- 요르단(히브리어 이름)?

개체 연결(Entity Linking, EL)은 각 멘션을 Wikidata, Wikipedia, DBpedia 또는 도메인별 지식 베이스(KB)의 고유 항목으로 해결합니다. 두 가지 하위 작업:

1. **후보 생성.** "Jordan"이 주어졌을 때, 어떤 KB 항목이 가능성이 있을까요?
2. **의미 명확화.** 문맥을 고려할 때, 어떤 후보가 정답일까요?

두 단계 모두 학습 가능합니다. 두 단계 모두 벤치마크가 존재합니다. 전체 파이프라인은 10년 동안 안정적이었지만, 변화하는 것은 의미 명확화기의 성능입니다.

## 개념

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**후보 생성.** 주어진 언급 표면 형태("Jordan")에 대해 별칭 인덱스에서 후보를 조회합니다. 위키피디아 별칭 사전은 대부분의 명명 엔티티를 커버합니다: "JFK" → John F. Kennedy, Jacqueline Kennedy, JFK 공항, JFK (영화). 일반적인 인덱스는 언급당 10-30개의 후보를 반환합니다.

**의미 명확화: 세 가지 접근법.**

1. **사전 확률 + 문맥 (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. 잘 작동하며, 빠르고 훈련이 필요 없습니다.
2. **임베딩 기반 (ESS / REL / Blink).** 언급 + 문맥을 인코딩합니다. 각 후보의 설명을 인코딩합니다. 코사인 유사도 최대값을 선택합니다. 2020-2024년 기본 접근법입니다.
3. **생성형 (GENRE, 2021; LLM 기반, 2023+).** 엔티티의 표준 이름을 토큰 단위로 디코딩합니다. 유효한 엔티티 이름 트라이(trie)로 제한되어 출력이 반드시 유효한 KB ID가 됩니다.

**엔드투엔드 vs 파이프라인.** 현대 모델(ELQ, BLINK, ExtEnD, GENRE)은 NER + 후보 생성 + 의미 명확화를 한 번에 실행합니다. 파이프라인 시스템은 구성 요소를 교체할 수 있기 때문에 여전히 프로덕션에서 우세합니다.

### 두 가지 측정 지표

- **언급 재현율 (후보 생성).** 정답 언급 중 후보 목록에 올바른 KB 항목이 나타나는 비율. 전체 파이프라인의 하한선입니다.
- **의미 명확화 정확도 / F1.** 올바른 후보가 주어졌을 때, 상위 1위 항목이 정답인 비율.

항상 둘 다 보고해야 합니다. 80% 후보 재현율에서 99% 의미 명확화 정확도를 가진 시스템은 80% 파이프라인 성능을 가집니다.

## 구축 방법

### 1단계: 위키피디아 리다이렉트로부터 별칭 인덱스 구축

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

위키피디아 별칭 데이터: ~18M (별칭, 엔티티) 쌍. Wikidata 덤프에서 다운로드. 역색인으로 저장.

### 2단계: 컨텍스트 기반 의미 구분

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

자카드 유사도는 간단한 예시입니다. 임베딩 기반 코사인 유사도로 대체 가능(`code/main.py` 2단계 참조).

### 3단계: 임베딩 기반 (BLINK 스타일)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

인덱스 생성 시 모든 KB 엔티티를 한 번 임베딩. 쿼리 시 언급 + 컨텍스트를 임베딩한 후 후보 풀과 내적 계산, 최대값 선택.

### 4단계: 생성형 엔티티 연결 (개념)

GENRE는 엔티티의 위키피디아 제목을 문자 단위로 디코딩. 제약 디코딩(20강 참조)을 통해 유효한 제목만 출력. KB 기반 트라이와 통합. 현대적 후속으로 REL-GEN과 구조화된 출력을 위한 LLM 프롬프트 EL이 있음.

```python
prompt = f"""Text: {text}
Mention: {mention}
이 언급에 가장 적합한 위키피디아 제목을 나열하세요.
JSON 형식으로 응답: {{"title": "..."}}"""
```

화이트리스트(선택지 개요)와 결합하면 2026년에 가장 쉽게 배포할 수 있는 EL 파이프라인.

### 5단계: AIDA-CoNLL에서 평가

AIDA-CoNLL은 표준 EL 벤치마크: 1,393개 로이터 기사, 34k 언급, 위키피디아 엔티티. 인-KB 정확도(`P@1`)와 아웃오브KB NIL 감지율 보고.

## 함정(Pitfalls)

- **NIL 처리.** 일부 멘션은 KB에 없음(신흥 엔티티, 잘 알려지지 않은 인물). 시스템은 잘못된 엔티티를 추측하는 대신 NIL을 예측해야 함. 별도로 측정됨.
- **멘션 경계 오류.** 상위 NER이 부분 스팬("Bank of America"가 "Bank"로만 태깅)을 놓침. EL 재현율(recall) 감소.
- **인기 편향.** 학습된 시스템은 빈번한 엔티티를 과예측함. ML 논문에서 "Michael I. Jordan" 멘션은 종종 농구 선수 Jordan과 연결됨.
- **크로스링구얼 EL.** 중국어 텍스트의 멘션을 영어 위키피디아 엔티티에 매핑. 다국어 인코더 또는 번역 단계 필요.
- **KB 노후화.** 새로운 회사, 이벤트, 인물은 작년 위키피디아 덤프에 없음. 프로덕션 파이프라인에는 갱신 루프 필요.

## 사용 방법

2026년 스택:

| 상황 | 선택 |
|-----------|------|
| 일반 영어 + 위키피디아 | BLINK 또는 REL |
| 다국어, KB = 위키피디아 | mGENRE |
| LLM 친화적, 하루 몇 건의 언급 | 후보 목록 + 제약된 JSON으로 Prompt Claude/GPT-4 |
| 도메인 특화 KB (의료, 법률) | KB 인식 검색 기능이 있는 커스텀 BERT + 도메인 AIDA 스타일 세트에서 파인튜닝 |
| 극저지연 | 정확한 일치 사전만 (Milne-Witten 기준선) |
| 연구 SOTA | GENRE / ExtEnD / 생성형 LLM-EL |

2026년에 출시되는 프로덕션 패턴: NER → 코레퍼런스 해결 → 각 언급에 대한 EL → 클러스터 축소(클러스터당 하나의 표준 엔티티로). 출력: 문서 내 엔티티당 하나의 KB ID, 언급당 하나가 아님.

## Ship It

`outputs/skill-entity-linker.md`로 저장:

```markdown
---
name: entity-linker
description: 엔티티 연결 파이프라인 설계 — 지식 베이스(KB), 후보 생성기, 구분자(disambiguator), 평가.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

사용 사례(도메인 KB, 언어, 처리량, 지연 시간 예산)가 주어졌을 때 다음을 출력:

1. 지식 베이스. Wikidata / Wikipedia / 커스텀 KB. 버전 날짜. 갱신 주기.
2. 후보 생성기. 별칭 인덱스(alias-index), 임베딩(embedding), 또는 하이브리드. 목표 멘션 재현율(mention recall) @ K.
3. 구분자(disambiguator). 사전 확률(prior) + 문맥(context), 임베딩 기반, 생성형(generative), 또는 LLM 프롬프트 기반.
4. NIL 전략. 상위 점수 임계값(threshold), 분류기(classifier), 또는 명시적 NIL 후보.
5. 평가. 멘션 재현율 @ 30, 상위-1 정확도(top-1 accuracy), NIL 감지 F1 점수(held-out set 기준).

멘션 재현율 기준(mention-recall baseline) 없이 EL 파이프라인을 거부(후보 생성기가 올바른 엔티티를 노출했는지 모르면 구분자 평가가 불가능). 유효한 KB ID에 대한 제약 출력(constrained output) 없이 LLM 프롬프트 기반 EL을 사용하는 파이프라인을 거부. 도메인 파인튜닝(fine-tuning) 없이 인기 편향(popularity bias)이 소수 엔티티(예: 이름 충돌)에 영향을 주는 시스템을 플래그 처리.
```

## 연습 문제

1. **쉬움.** `code/main.py`에 10개의 모호한 언급(Paris, Jordan, Apple)에 대한 사전+문맥 기반 구분기를 구현하세요. 올바른 엔티티를 수동으로 라벨링하세요. 정확도를 측정하세요.
2. **중간.** 50개의 모호한 언급을 문장 변환기로 인코딩하세요. 각 후보의 설명을 임베딩하세요. 임베딩 기반 구분과 Jaccard 문맥 중첩을 비교하세요.
3. **어려움.** 1k-엔티티 도메인 KB(예: 회사 내 직원 + 제품)를 구축하세요. NER + EL을 엔드투엔드로 구현하세요. 100개의 홀드아웃 문장에 대해 정밀도와 재현율을 측정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 엔티티 링킹(Entity linking, EL) | 위키피디아에 링크 | 멘션(mention)을 고유한 KB 항목으로 매핑. |
| 후보 생성(Candidate generation) | 누구일까? | 멘션에 대한 타당한 KB 항목들의 후보 목록 반환. |
| 의미 구분(Disambiguation) | 올바른 것 선택 | 문맥을 사용해 후보들의 점수를 매기고 최종 선택. |
| 별칭 인덱스(Alias index) | 조회 테이블 | 표면 형태(surface form) → 후보 엔티티 매핑. |
| NIL | KB에 없음 | 어떤 KB 항목과도 일치하지 않는다는 명시적 예측. |
| KB(Knowledge base) | 지식 베이스 | Wikidata, Wikipedia, DBpedia 또는 도메인 특화 KB. |
| AIDA-CoNLL | 벤치마크 | 1,393개의 로이터 기사와 정답 엔티티 링크. |

## 추가 자료

- [Milne, Witten (2008). 위키피디아를 이용한 링크 학습](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 사전+컨텍스트 접근법의 기반 연구.
- [Wu et al. (2020). 밀집 엔티티 검색을 통한 제로샷 엔티티 링킹 (BLINK)](https://arxiv.org/abs/1911.03814) — 임베딩 기반 핵심 연구.
- [De Cao et al. (2021). 자기회귀식 엔티티 검색 (GENRE)](https://arxiv.org/abs/2010.00904) — 제약 디코딩을 활용한 생성형 엔티티 링킹.
- [Hoffart et al. (2011). 텍스트 내 명명된 엔티티의 강건한 의미역 결정 (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 벤치마크 논문.
- [REL: 거인의 어깨 위에 선 엔티티 링커 (2020)](https://arxiv.org/abs/2006.01969) — 오픈 프로덕션 스택.