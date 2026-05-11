# 자연어 추론 — 텍스트 함의

> "t entails h"는 인간이 t를 읽고 h가 참이라고 결론짓는 것을 의미합니다. NLI는 함의 / 모순 / 중립을 예측하는 작업입니다. 표면적으로는 지루하지만, 실제 운영에서는 핵심 역할을 합니다.

**유형:** 학습
**언어:** Python
**선수 지식:** Phase 5 · 05 (감정 분석), Phase 5 · 13 (질문 응답)
**소요 시간:** ~60분

## 문제 정의

요약기를 구축했습니다. 요약기가 요약을 생성했습니다. 요약에 환각(hallucination)이 포함되지 않았다는 것을 어떻게 알 수 있을까요?

챗봇을 구축했습니다. 챗봇이 "예"라고 답변했습니다. 해당 답변이 검색된 문서(passage)에 의해 뒷받침된다는 것을 어떻게 알 수 있을까요?

10,000개의 뉴스 기사를 주제별로 분류해야 합니다. 학습 레이블이 없습니다. 모델을 재사용할 수 있을까요?

이 세 가지 문제는 모두 자연어 추론(Natural Language Inference, NLI)으로 귀결됩니다. NLI는 다음과 같은 질문을 합니다: 전제(premise) `t`와 가설(hypothesis) `h`가 주어졌을 때, `h`가 `t`에 의해 함축(entailment)되는지, 모순(contradiction)되는지, 아니면 중립(neutral, 무관)인지?

- **환각 검사:** `t` = 원본 문서, `h` = 요약 주장. 함축되지 않음 = 환각.
- **근거 기반 QA:** `t` = 검색된 문서, `h` = 생성된 답변. 함축되지 않음 = 허위 생성(fabrication).
- **제로샷 분류:** `t` = 문서, `h` = 언어화된 레이블("이것은 스포츠에 관한 것입니다"). 함축 = 예측된 레이블.

하나의 작업, 세 가지 프로덕션 활용 사례. 이것이 모든 RAG(Retrieval-Augmented Generation) 평가 프레임워크가 내부적으로 NLI 모델을 포함하는 이유입니다.

## 개념

![NLI: 세 가지 분류, 전제 vs 가설](../assets/nli.svg)

**세 가지 라벨.**

- **함의(Entailment).** `t` → `h`. "고양이가 매트 위에 있다"는 "고양이가 있다"를 함의합니다.
- **모순(Contradiction).** `t` → ¬`h`. "고양이가 매트 위에 있다"는 "고양이가 없다"와 모순됩니다.
- **중립(Neutral).** 어느 쪽으로도 추론 불가. "고양이가 매트 위에 있다"는 "고양이가 배고프다"와 중립입니다.

**논리적 함의가 아님.** NLI는 *자연어* 추론입니다 — 엄격한 논리가 아닌 일반 인간 독자가 추론할 수 있는 내용을 다룹니다. "John이 그의 개를 산책시켰다"는 NLI에서 "John은 개를 가지고 있다"를 함의하지만, 엄격한 1차 논리에서는 소유를 공리로 명시해야만 인정됩니다.

**데이터셋.**

- **SNLI** (2015). 570k개의 인간 주석 쌍, 이미지 캡션을 전제로 사용. 좁은 도메인.
- **MultiNLI** (2017). 10개 장르에 걸친 433k개 쌍. 2026년 표준 학습 코퍼스.
- **ANLI** (2019). 적대적 NLI. 인간이 기존 모델을 깨기 위해 특별히 설계한 예시. 더 어려움.
- **DocNLI, ConTRoL** (2020–21). 문서 길이 전제. 다중 홉 및 장거리 추론 테스트.

**아키텍처.** 트랜스포머 인코더(BERT, RoBERTa, DeBERTa)가 `[CLS] 전제 [SEP] 가설 [SEP]`을 읽습니다. `[CLS]` 표현은 3-way 소프트맥스로 입력됩니다. MNLI로 학습하고, 보유된 벤치마크에서 평가하며, 동일 분포 쌍에서 90%+ 정확도를 달성합니다.

**NLI를 통한 제로샷.** 문서와 후보 라벨이 주어졌을 때, 각 라벨을 가설로 변환("이 텍스트는 스포츠에 관한 것이다"). 각각에 대한 함의 확률을 계산합니다. 최댓값을 선택합니다. 이것이 Hugging Face의 `zero-shot-classification` 파이프라인의 메커니즘입니다.

## 구축 방법

### 1단계: 사전 학습된 NLI 모델 실행

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # 모든 라벨 반환; deprecated된 return_all_scores=True 대체

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

프로덕션 NLI의 경우 `facebook/bart-large-mnli`와 `microsoft/deberta-v3-large-mnli`가 오픈 기본값입니다. DeBERTa-v3는 리더보드에서 1위를 차지합니다.

### 2단계: 제로샷 분류

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

템플릿은 기본적으로 "This example is about {label}."입니다. `hypothesis_template`로 커스터마이징할 수 있습니다. 학습 데이터 불필요. 파인튜닝 없음. 바로 사용 가능.

### 3단계: RAG 신뢰성 검증

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

이것은 RAGAS 신뢰성의 핵심입니다. 생성된 답변을 원자적 주장으로 분할합니다. 각 주장을 검색된 컨텍스트와 대조합니다. 함의하는 비율을 보고합니다.

### 4단계: 직접 구현한 NLI 분류기 (개념적)

`code/main.py`에서 stdlib 전용 토이를 참조하세요: 전제와 가설은 어휘적 중복 + 부정 감지를 통해 비교됩니다. 트랜스포머 모델과 경쟁할 수준은 아니지만, 작업 형태를 보여줍니다: 두 텍스트 입력, 3가지 라벨 출력, 손실 = `{entail, contradict, neutral}`에 대한 교차 엔트로피.

## 함정

- **가설 전용 단축 경로.** 모델은 SNLI에서 가설만으로 레이블을 ~60% 정확도로 예측할 수 있습니다. "not", "nobody", "never"와 같은 단어가 모순(contradiction)과 강한 상관관계를 보이기 때문입니다. 레이블 누출(label leakage) 감지를 위한 강력한 기준선입니다.
- **어휘 중복 휴리스틱.** "모든 부분 수열은 함축된다"는 부분 수열 휴리스틱(subsequence heuristic)은 SNLI에서는 통과하지만 HANS/ANLI에서는 실패합니다. 적대적 벤치마크(adversarial benchmarks)를 사용하세요.
- **문서 길이 성능 저하.** 단일 문장 NLI 모델은 문서 길이 전제(document-length premises)에서 20+ F1 점수가 하락합니다. 긴 컨텍스트에는 DocNLI로 훈련된 모델을 사용하세요.
- **제로샷 템플릿 민감도.** "This example is about {label}" vs "{label}" vs "The topic is {label}"과 같은 템플릿은 정확도를 10% 이상 변동시킬 수 있습니다. 템플릿을 조정하세요.
- **도메인 불일치.** MNLI는 일반 영어 텍스트로 훈련됩니다. 법률, 의료, 과학 텍스트에는 도메인 특화 NLI 모델(예: SciNLI, MedNLI)이 필요합니다.

## 사용 방법

2026 스택:

| 사용 사례 | 모델 |
|---------|-------|
| 일반 목적 NLI | `microsoft/deberta-v3-large-mnli` |
| 빠른/엣지 | `cross-encoder/nli-deberta-v3-base` |
| 제로샷 분류(가벼운) | `facebook/bart-large-mnli` |
| 문서 수준 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 다국어 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG에서의 환각 탐지 | RAGAS/DeepEval 내부의 NLI 레이어 |

2026 메타 패턴: NLI는 텍스트 이해의 덕트 테이프입니다. "A가 B를 지지하는가?" 또는 "A가 B와 모순되는가?"와 같은 질문이 필요할 때마다 다른 LLM 호출을 하기 전에 NLI를 먼저 고려하세요.

## Ship It

`outputs/skill-nli-picker.md`로 저장:

```markdown
---
name: nli-picker
description: 분류 / 충실도 / 제로샷 작업을 위한 NLI 모델, 라벨 템플릿, 평가 설정을 선택합니다.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

사용 사례(충실도 검증, 제로샷 분류, 문서 수준 추론)가 주어졌을 때 다음을 출력합니다:

1. 모델. 명명된 NLI 체크포인트. 도메인, 길이, 언어와의 연관성 근거.
2. 템플릿(제로샷인 경우). 언어화 패턴. 예시.
3. 임계값. 결정 규칙을 위한 함의(entailment) 절단값. 보정(calibration) 기반 근거.
4. 평가. 보유된 레이블 세트에서의 정확도, 가설 전용 베이스라인, 적대적 부분집합.

100개 예시 레이블 검증 없이 제로샷 분류를 출하하는 것을 거부합니다. 문장 수준 NLI 모델을 문서 길이 전제에 사용하는 것을 거부합니다. NLI가 환각(hallucination)을 해결한다는 주장을 플래그 처리합니다 — NLI는 환각을 줄일 뿐, 완전히 제거하지는 않습니다.
```

## 연습 문제

1. **쉬움.** 3개 클래스(클래스)를 모두 포함하는 20개의 수작업 (전제, 가설, 레이블) 삼중항에 대해 `facebook/bart-large-mnli`를 실행합니다. 정확도를 측정합니다. 적대적 "부분 수열 휴리스틱" 트랩("I did not eat the cake" vs "I ate the cake")을 추가하고 모델이 실패하는지 확인합니다.
2. **중간.** 100개의 AG News 헤드라인에서 제로샷 템플릿 `"This text is about {label}"`을 `"The topic is {label}"` 및 `"{label}"`과 비교합니다. 정확도 변동을 보고합니다.
3. **어려움.** RAG 충실도 검사기 구축: 원자적 주장 분해 + 주장별 NLI. 골드 컨텍스트가 있는 50개의 RAG 생성 답변에 대해 평가합니다. 수작업 레이블 대비 거짓 양성 및 거짓 음성 비율을 측정합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|----------|
| NLI | 자연어 추론(Natural Language Inference) | 전제-가설 관계의 3-way 분류. |
| RTE | 텍스트 함의 인식(Recognizing Textual Entailment) | NLI의 이전 명칭; 동일한 작업. |
| 함의(Entailment) | "t가 h를 함의한다" | 일반적인 독자는 t가 주어졌을 때 h가 참이라고 결론짓는다. |
| 모순(Contradiction) | "t가 h를 배제한다" | 일반적인 독자는 t가 주어졌을 때 h가 거짓이라고 결론짓는다. |
| 중립(Neutral) | "결정 불가" | t에서 h로의 추론이 어느 쪽도 성립하지 않는다. |
| 제로샷 분류(Zero-shot classification) | NLI를 분류기로 사용 | 레이블을 가설로 변환, 최대 함의를 선택. |
| 충실도(Faithfulness) | 답변이 지원되는가? | (검색된 문맥, 생성된 답변)에 대한 NLI. |

## 추가 자료

- [Bowman et al. (2015). 자연어 추론을 위한 대규모 주석 코퍼스](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). 추론을 통한 문장 이해를 위한 광범위한 도전 코퍼스](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). 적대적 자연어 추론](https://arxiv.org/abs/1910.14599) — ANLI 벤치마크.
- [Yin, Hay, Roth (2019). 제로샷 텍스트 분류 벤치마킹](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [He et al. (2021). DeBERTa: 분리된 어텐션 기반 디코딩 강화 BERT](https://arxiv.org/abs/2006.03654) — 2026 NLI 작업용 모델.