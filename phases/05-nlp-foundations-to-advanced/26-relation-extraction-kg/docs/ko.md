# 관계 추출 & 지식 그래프 구축

> NER은 엔티티를 찾았습니다. 엔티티 연결(Linking)은 이를 고정했습니다. 관계 추출은 이들 사이의 엣지(edge)를 찾습니다. 지식 그래프는 노드, 엣지, 그리고 그 출처(provenance)의 총합입니다.

**유형:** 구축(Build)
**언어:** Python
**사전 요구 사항:** Phase 5 · 06 (NER), Phase 5 · 25 (엔티티 연결)
**소요 시간:** ~60분

## 문제 정의

분석가는 "Tim Cook은 2011년에 Apple의 CEO가 되었다"라는 문장을 읽습니다. 네 가지 사실:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

관계 추출(Relation Extraction, RE)은 자유 텍스트를 구조화된 트리플 `(주어, 관계, 목적어)`로 변환합니다. 코퍼스 전체에 걸쳐 집계하면 지식 그래프가 생성됩니다. 이를 집계하고 쿼리하면 RAG, 분석 또는 규정 준수 감사를 위한 추론 기반이 됩니다.

2026년 문제: LLM은 열정적으로 관계를 추출합니다. 지나치게 열정적입니다. 소스 텍스트가 지원하지 않는 트리플을 환각(hallucination)합니다. 출처(provenance)가 없으면 실제 트리플과 그럴듯한 허구를 구분할 수 없습니다. 2026년 해결책은 AEVS 스타일의 앵커-검증(anchor-and-verify) 파이프라인입니다.

## 개념

![텍스트 → 트리플 → 지식 그래프](../assets/relation-extraction.svg)

**트리플 형식.** `(주어_개체, 관계_유형, 목적어_개체)`. 관계는 닫힌 온톨로지(Wikidata 속성, FIBO, UMLS) 또는 열린 집합(OpenIE 스타일, 모든 관계 허용)에서 비롯됩니다.

**세 가지 추출 접근법.**

1. **규칙/패턴 기반.** 히어스트 패턴: "X such as Y" → `(Y, isA, X)`. 수작업 정규 표현식 추가. 취약하지만 정확하고 설명 가능.
2. **지도 학습 분류기.** 문장 내 두 개체 멘션이 주어졌을 때, 고정된 집합에서 관계를 예측. TACRED, ACE, KBP로 학습. 2015–2022년 표준 방식.
3. **생성형 LLM.** 모델에 트리플 생성을 프롬프트. 즉시 작동. 출처 필요 시, 그럴듯한 허구를 생성.

**AEVS(Anchor-Extraction-Verification-Supplement, 2026).** 현재 환각 완화 프레임워크:

- **앵커.** 모든 개체 스팬과 관계 구문 스팬을 정확한 위치와 함께 식별.
- **추출.** 앵커 스팬에 연결된 트리플 생성.
- **검증.** 각 트리플 요소를 소스 텍스트와 대조; 지원되지 않는 항목 거부.
- **보완.** 커버리지 패스로 앵커 스팬 누락 방지.

환각이 급격히 감소. 더 많은 계산 리소스 필요하나 감사 가능.

**열린 집합 vs 닫힌 집합 트레이드오프.**

- **닫힌 온톨로지.** 고정된 속성 목록(예: Wikidata의 11,000+ 속성). 예측 가능. 쿼리 가능. 새로운 관계 생성 어려움.
- **열린 IE.** 모든 구문이 관계가 됨. 높은 재현율. 낮은 정밀도. 쿼리 시 혼란.

프로덕션 KG는 일반적으로 혼합 사용: 발견을 위한 열린 IE, 이후 닫힌 온톨로지에 관계를 정규화하여 메인 그래프에 병합.

## 구축 방법

### 1단계: 패턴 기반 추출

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

전체 토이 추출기는 `code/main.py`를 참조하세요. Hearst 패턴은 디버깅이 가능하기 때문에 여전히 도메인 특화 파이프라인에 사용됩니다.

### 2단계: 지도식 관계 분류

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL은 시퀀스-투-시퀀스 관계 추출기입니다: 텍스트를 입력하면 Wikidata 속성 ID로 트리플이 출력됩니다. 원격 감독 데이터로 파인튜닝되었으며 표준 오픈 가중치 기준선입니다.

### 3단계: 앵커링 기반 LLM 프롬프트 추출

```python
prompt = f"""텍스트에서 (주어, 관계, 목적어) 트리플을 추출하세요.
각 트리플에 대해 소스 텍스트에서 정확한 문자 범위를 포함하세요.

텍스트: {text}

출력 JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

텍스트에 완전히 지원되는 트리플만 포함하세요. 명시된 내용 이상의 추론은 하지 마세요.
"""
```

반환된 모든 범위를 소스와 대조하여 검증하세요. `text[start:end] != triple_entity`인 경우 거부하세요. 이는 AEVS "검증" 단계의 최소 형태입니다.

### 4단계: 폐쇄형 온톨로지로 정규화

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (주어/목적어 반전)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # 매핑되지 않은 개방형 관계 삭제 또는 수동 검토로 전달
```

정규화는 종종 엔지니어링 작업의 60-80%를 차지합니다. 이에 대한 예산을 확보하세요.

### 5단계: 소규모 그래프 구축 및 쿼리

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

이것은 모든 RAG-over-KG 시스템의 원자입니다. RDF 트리플 스토어(Blazegraph, Virtuoso), 속성 그래프(Neo4j) 또는 벡터 증강 그래프 스토어로 확장하세요.

## 주의 사항

- **공참조 해결 전 관계 추출(RE).** "He founded Apple" — RE는 "he"가 누구인지 알아야 합니다. 공참조 해결(coref)을 먼저 수행하세요(레슨 24).
- **엔티티 정규화.** "Apple Inc"와 "Apple"은 동일한 노드로 해결되어야 합니다. 엔티티 연결(linking)을 먼저 수행하세요(레슨 25).
- **환각 트리플.** LLM은 텍스트가 지원하지 않는 트리플을 생성합니다. 스팬 검증을 강제 적용하세요.
- **관계 정규화 드리프트.** 개방형 정보 추출(Open IE) 관계는 불일치합니다("was born in", "came from", "is a native of"). 정규화된 ID로 축소하지 않으면 그래프를 쿼리할 수 없습니다.
- **시간적 오류.** "Tim Cook is CEO of Apple" — 현재는 참이지만 2005년에는 거짓입니다. 많은 관계는 시간 제약이 있습니다. 한정자(Wikidata의 `P580` 시작 시간, `P582` 종료 시간)를 사용하세요.
- **도메인 불일치.** REBEL은 위키피디아로 학습되었습니다. 법률, 의료, 과학 텍스트는 도메인 특화 미세 조정(fine-tuning)된 RE 모델이 필요한 경우가 많습니다.

## 사용 방법

2026 스택:

| 상황 | 선택 |
|-----------|------|
| 빠른 프로덕션, 일반 도메인 | REBEL 또는 Wikidata 정규화(Canonicalization)를 사용한 LlamaPred |
| 도메인 특화(생명과학, 법률) | SciREX 스타일 도메인 파인튜닝(Fine-tuning) + 사용자 정의 온톨로지(Ontology) |
| LLM 프롬프트 기반, 검증된 출력 | AEVS 파이프라인: 앵커(Anchor) → 추출(Extract) → 검증(Verify) → 보완(Supplement) |
| 대량 뉴스 정보 추출(IE) | 패턴 기반 + 지도 학습 하이브리드 |
| 처음부터 KG 구축 | Open IE + 수동 정규화(Canonicalization) 단계 |
| 시간 기반 KG | 한정자(시작/종료 시간, 특정 시점) 포함 추출 |

통합 패턴: 개체명 인식(NER) → 공동 참조 해결(Coref) → 개체 연결(Entity Linking) → 관계 추출(Relation Extraction) → 온톨로지 매핑(Ontology Mapping) → 그래프 로드(Graph Load). 모든 단계는 잠재적 품질 게이트(Quality Gate)가 될 수 있습니다.

## Ship It

`outputs/skill-re-designer.md`로 저장:

```markdown
---
name: re-designer
description: 출처(provenance)와 정규화(canonicalization)를 갖춘 관계 추출 파이프라인을 설계합니다.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

주어진 말뭉치(도메인, 언어, 규모)와 다운스트림 사용 사례(KG-RAG, 분석, 규정 준수)에 따라 다음을 출력합니다:

1. 추출기(Extractor). 패턴 기반 / 지도 학습 / LLM / AEVS 하이브리드. 정밀도(precision) 대 재현율(recall) 목표에 따른 근거.
2. 온톨로지(Ontology). 폐쇄적 속성 목록(Wikidata / 도메인) 또는 정규화(canonicalization) 패스가 포함된 개방형 IE.
3. 출처(Provenance). 모든 트리플은 소스 문자 스팬(char-span) + 문서 ID를 포함. 감사(audit)를 위해 필수.
4. 병합 전략(Merge strategy). 정규화된 엔티티 ID + 관계 ID + 시간 한정자(temporal qualifiers); 중복 제거(dedup) 정책.
5. 평가(Evaluation). 200개의 수작업 라벨링된 트리플에 대한 정밀도/재현율 + LLM 추출 샘플의 환각(hallucination) 비율.

스팬 검증(source provenance) 없이 LLM 기반 관계 추출 파이프라인을 거부합니다. 정규화 없이 개방형 IE 출력이 프로덕션 그래프로 유입되는 것을 거부합니다. 시간 한정 관계(고용주, 배우자, 직위)에 시간 한정자가 없는 파이프라인을 플래그 처리합니다.
```

## 연습 문제

1. **쉬움.** `code/main.py`에 있는 패턴 추출기를 5개의 뉴스 기사 문장에 실행해 보세요. 정밀도(precision)를 수동으로 확인하세요.
2. **중간.** 동일한 문장에 REBEL(또는 소형 LLM)을 사용해 보세요. 트리플(triple)을 비교하세요. 어떤 추출기가 더 높은 정밀도(precision)를 가지나요? 더 높은 재현율(recall)을 가지나요?
3. **어려움.** AEVS 파이프라인을 구축하세요: LLM으로 추출 + 소스(source) 대비 스팬(span) 검증. 50개의 위키피디아 스타일 문장에서 검증 단계 전후의 환각(hallucination) 비율을 측정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 트리플(Triple) | 주어-관계-목적어(Subject-relation-object) | KG의 원자적 단위인 `(s, r, o)` 튜플. |
| 오픈 정보 추출(Open IE) | 아무거나 추출(Extract anything) | 개방형 어휘 관계 구문; 높은 재현율, 낮은 정밀도. |
| 폐쇄 온톨로지(Closed ontology) | 고정 스키마(Fixed schema) | 제한된 관계 유형 집합(Wikidata, UMLS, FIBO). |
| 캐노니컬화(Canonicalization) | 모든 것 정규화(Normalize everything) | 표면 이름/관계를 정규 ID로 매핑. |
| AEVS | 근거 기반 추출(Grounded extraction) | 앵커-추출-검증-보완 파이프라인(2026). |
| 프로비넌스(Provenance) | 출처 연결(Source-of-truth link) | 모든 트리플은 문서 ID + 문자 스팬을 소스로 포함. |
| 원격 감독(Distant supervision) | 저렴한 라벨(Cheap labels) | 텍스트와 기존 KG를 정렬하여 훈련 데이터 생성.

## 추가 자료

- [Mintz et al. (2009). 레이블 없는 데이터를 이용한 관계 추출을 위한 원격 감독(distant supervision)](https://www.aclweb.org/anthology/P09-1113.pdf) — 원격 감독(distant supervision) 논문.
- [Huguet Cabot, Navigli (2021). REBEL: 종단간 언어 생성(end-to-end language generation)을 통한 관계 추출(relation extraction)](https://aclanthology.org/2021.findings-emnlp.204.pdf) — 시퀀스 투 시퀀스(seq2seq) 관계 추출 핵심 연구.
- [Wadden et al. (2019). 컨텍스트화된 스팬 표현(contextualized span representations)을 이용한 개체, 관계, 이벤트 추출(DyGIE++)](https://arxiv.org/abs/1909.03546) — 통합 정보 추출(joint IE).
- [AEVS — 앵커 추출-검증-보완(Anchor-Extraction-Verification-Supplement) 프레임워크](https://www.mdpi.com/2073-431X/15/3/178) — 2026년 환각(hallucination) 완화 설계.
- [Wikidata SPARQL 튜토리얼](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 표준 그래프 쿼리.