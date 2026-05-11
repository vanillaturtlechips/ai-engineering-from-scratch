# 기계 번역

> 번역은 30년 동안 NLP 연구를 후원한 작업이며, 지금도 계속 후원하고 있습니다.

**유형:** 구축
**언어:** Python
**선수 지식:** Phase 5 · 10 (어텐션 메커니즘), Phase 5 · 04 (GloVe, FastText, 서브워드)
**소요 시간:** ~75분

## 문제 정의

모델이 한 언어의 문장을 읽고 다른 언어의 문장을 생성합니다. 길이는 다양합니다. 단어 순서도 다양합니다. 일부 소스 단어는 여러 타겟 단어에 매핑되고 그 반대도 마찬가지입니다. 관용구는 일대일 매핑을 거부합니다. 프랑스어로 "I miss you"는 "tu me manques"인데, 직역하면 "you are lacking to me"입니다. 이 경우 단어 수준의 정렬은 살아남지 못합니다.

기계 번역은 NLP가 인코더-디코더(encoder-decoder), 어텐션(attention), 트랜스포머(transformer), 그리고 결국 전체 LLM 패러다임을 발명하도록 강제했던 과제입니다. 모든 진전은 번역 품질이 측정 가능했고 인간과 기계 사이의 격차가 완고했기 때문에 이루어졌습니다.

이 레슨은 역사 수업을 건너뛰고 2026년의 작동 파이프라인을 가르칩니다: 사전 훈련된 다국어 인코더-디코더(NLLB-200 또는 mBART), 서브워드 토큰화(subword tokenization), 빔 서치(beam search), BLEU 및 chrF 평가, 그리고 여전히 감지되지 않은 채 프로덕션으로 배송되는 소수의 실패 모드들입니다.

## 개념

![MT 파이프라인: 토크나이징 → 인코딩 → 어텐션 디코딩 → 디토크나이징](../assets/mt-pipeline.svg)

현대 기계 번역(MT)은 병렬 텍스트로 학습된 트랜스포머 인코더-디코더 모델입니다. 인코더는 소스 언어의 토큰화 방식으로 입력을 읽습니다. 디코더는 크로스-어텐션(레슨 10)을 통해 인코더 출력을 활용해 대상 언어를 서브워드 단위로 생성하며, 빔 서치를 사용해 탐욕적 디코딩의 함정을 피합니다. 출력은 디토크나이징, 디트루케이싱 처리되며 참조 번역과 비교 평가됩니다.

실제 MT 품질을 좌우하는 세 가지 운영 선택 사항이 있습니다.

- **토크나이저.** 다국어 코퍼스로 학습된 SentencePiece BPE. 언어 간 공유 어휘집이 NLLB에서 제로샷 언어 쌍을 가능하게 합니다.
- **모델 크기.** NLLB-200 600M 파라미터는 노트북에 탑재 가능합니다. NLLB-200 3.3B는 공개된 기본 프로덕션 모델입니다. 54.5B는 연구용 상한선입니다.
- **디코딩.** 일반 콘텐츠에는 빔 너비 4-5를 사용합니다. 너무 짧은 출력을 방지하기 위해 길이 페널티를 적용합니다. 용어 일관성이 필요할 때는 제약 디코딩을 사용합니다.

## 구축 방법

### 1단계: 사전 훈련된 MT 모델 호출

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

여기서 세 가지 사항이 중요합니다. `src_lang`은 토크나이저에 적용할 스크립트와 분할 방식을 알려줍니다. `forced_bos_token_id`는 디코더에 생성할 언어를 지정합니다. 둘 다 NLLB 전용 기법이며, mBART와 M2M-100은 자체 관례를 사용하며 서로 교환 불가능합니다.

### 2단계: BLEU와 chrF

BLEU는 출력과 참조 간 n-그램 중첩을 측정합니다. 1-4개의 참조 n-그램 크기, 정밀도의 기하 평균, 너무 짧은 출력에 대한 간결성 페널티를 적용합니다. 점수는 [0, 100] 범위입니다. 일반적으로 사용됩니다. 해석이 어려울 수 있습니다: 30 BLEU는 "사용 가능", 40은 "양호", 50은 "탁월"하며, 1 BLEU 미만의 차이는 노이즈입니다.

chrF는 문자 수준 F-점수를 측정합니다. BLEU가 일치 항목을 과소 계산하는 형태학적으로 풍부한 언어에 더 민감합니다. BLEU와 함께 보고되는 경우가 많습니다.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

항상 `sacrebleu`를 사용하세요. 토큰화를 정규화하여 논문 간 점수를 비교할 수 있습니다. 직접 BLEU 계산을 구현하면 오해의 소지가 있는 벤치마크가 발생할 수 있습니다.

### 3단계: 3단계 평가 체계 (2026)

현대 MT 평가는 세 가지 보완적 메트릭 패밀리를 사용합니다. 최소 두 가지를 함께 사용하세요.

- **휴리스틱** (BLEU, chrF). 빠르고 참조 기반이며 해석이 용이하지만 패러프레이즈에 둔감합니다. 레거시 비교 및 회귀 감지에 사용합니다.
- **학습 기반** (COMET, BLEURT, BERTScore). 인간 판단으로 훈련된 신경망 모델; 번역과 소스 및 참조의 의미적 유사성을 비교합니다. COMET은 2023년 이후 MT 연구와 가장 높은 연관성을 가지며, 품질이 중요한 2026년 프로덕션 기본값입니다.
- **LLM 평가자** (참조 불필요). 대형 모델에 유창성, 적절성, 어조, 문화적 적절성을 평가하도록 프롬프트를 제공합니다. 평가 기준이 잘 설계된 경우 GPT-4 평가자는 인간 동의율과 약 80% 일치합니다. 참조가 없는 개방형 콘텐츠에 사용합니다.

실용적인 2026년 스택: BLEU와 chrF를 위한 `sacrebleu`, COMET을 위한 `unbabel-comet`, 최종 인간 대상 신호를 위한 프롬프트 LLM. 프로덕션 데이터에 신뢰하기 전에 모든 메트릭을 50-100개의 인간 레이블 예제로 보정하세요.

참조 불필요 메트릭(COMET-QE, BLEURT-QE, LLM 평가자)은 참조 번역이 없는 긴 꼬리 언어 쌍에 중요합니다.

### 4단계: 프로덕션에서 발생하는 문제

위의 작업 파이프라인은 80%의 경우 유창하게 번역하고 나머지 20%는 조용히 실패합니다. 명명된 실패 모드:

- **환각(Hallucination).** 모델이 소스에 없는 내용을 생성합니다. 익숙하지 않은 도메인 어휘에서 흔합니다. 증상: 출력은 유창하지만 소스에 명시되지 않은 사실을 주장합니다. 완화: 도메인 용어에 대한 제약 디코딩, 규제 콘텐츠에 대한 인간 검토, 입력보다 훨씬 긴 출력 모니터링.
- **오프타겟 생성.** 모델이 잘못된 언어로 번역합니다. NLLB는 희귀 언어 쌍에서 이 문제가 의외로 자주 발생합니다. 완화: `forced_bos_token_id` 확인 및 출력에 항상 언어 ID 모델 검사 적용.
- **용어 드리프트.** "Sign up"이 문서 1에서는 "s'inscrire", 문서 2에서는 "créer un compte"로 번역됩니다. UI 텍스트 및 사용자 노출 문자열에서는 원시 품질보다 일관성이 중요합니다. 완화: 용어집 제약 디코딩 또는 사후 편집 사전.
- **격식 수준 불일치.** 프랑스어 "tu" vs "vous", 일본어 경어 수준. 모델은 훈련에서 더 흔했던 형태를 선택합니다. 고객 대상 콘텐츠에서는 일반적으로 잘못됩니다. 완화: 모델이 지원하는 경우 격식 토큰으로 프롬프트 접두사 추가 또는 격식 전용 코퍼스로 소규모 모델 파인튜닝.
- **짧은 입력에서 길이 폭발.** 매우 짧은 입력 문장은 길이 페널티가 소스 토큰 ~5 미만에서 급격히 감소하기 때문에 과도하게 긴 번역을 생성하는 경우가 많습니다. 완화: 소스 길이에 비례하는 하드 최대 길이 제한.

### 5단계: 도메인 파인튜닝

사전 훈련된 모델은 일반주의자입니다. 법률, 의료 또는 게임 대화 번역은 도메인 병렬 데이터로 파인튜닝하면 측정 가능한 이점을 얻습니다. 레시피는 특별하지 않습니다:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

수천 개의 고품질 병렬 예제가 수십만 개의 노이즈가 많은 웹 스크래핑 예제보다 낫습니다. 훈련 데이터의 품질이 프로덕션에서 가장 큰 레버리지입니다.

## 사용 방법

2026년 기계 번역(MT) 생산 스택:

| 사용 사례 | 권장 시작점 |
|---------|---------------------------|
| 모든 언어 쌍, 200개 언어 | `facebook/nllb-200-distilled-600M` (노트북) 또는 `nllb-200-3.3B` (프로덕션) |
| 영어 중심, 고품질, 50개 언어 | `facebook/mbart-large-50-many-to-many-mmt` |
| 짧은 실행, 저렴한 추론, 영어-프랑스어/독일어/스페인어 | Helsinki-NLP / Marian 모델 |
| 지연 시간 민감 브라우저 측 | ONNX-양자화 Marian (~50 MB) |
| 최대 품질, 비용 지불 가능 | GPT-4 / Claude / Gemini 번역 프롬프트 사용 |

2026년 현재 LLM은 관용적 콘텐츠 및 긴 컨텍스트에서 여러 언어 쌍에 대해 전문 MT 모델보다 우수한 성능을 보입니다. 다만 토큰당 비용과 지연 시간이라는 트레이드오프가 존재합니다. 처리량보다 컨텍스트 길이, 스타일 일관성 또는 프롬프트를 통한 도메인 적응이 더 중요한 경우 LLM을 선택하세요.

## Ship It

`outputs/skill-mt-evaluator.md`로 저장:

```markdown
---
name: mt-evaluator
description: 기계 번역 출력을 배송 전 평가합니다.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

소스 텍스트와 후보 번역문이 주어졌을 때 다음을 출력합니다:

1. 자동 점수 추정치. 예상되는 BLEU(BLEU)와 chrF(chrF) 범위. 참조 번역(reference) 사용 가능 여부 명시.
2. 5가지 인간 검증 체크리스트: (a) 내용 보존(오류 생성 없음), (b) 올바른 언어 사용, (c) 문체/격식 일치도, (d) 제공된 용어집(glossary)과의 용어 일관성, (e) 생략 또는 길이 폭주 없음.
3. 도메인별 심층 검토 항목. 예: 법률 분야 - 인명/법률 조항 인용. 의료 분야 - 약물명 및 용량. UI 분야 - 플레이스홀더 변수 `{name}`.
4. 신뢰도 플래그. "Ship"(배송) / "Ship with review"(검토 후 배송) / "Do not ship"(배송 불가). 2단계에서 발견된 문제 심각도에 따라 결정.

출력물에 대한 언어 식별(Language-ID) 검사 없이 번역을 배송하지 않습니다. 사용자가 참조 번역 없이 평가(COMET-QE, BLEURT-QE)를 명시적으로 선택하지 않는 한 참조 번역 없이 평가하지 않습니다. 1000토큰을 초과하는 콘텐츠는 청크 분할 번역이 필요할 수 있음을 표시합니다.
```

## 연습 문제

1. **쉬움.** `nllb-200-distilled-600M`을 사용하여 5문장으로 구성된 영어 단락을 프랑스어로 번역한 후 다시 영어로 번역하세요. 원본과 왕복 번역 결과가 얼마나 유사한지 측정하세요. 의미 보존은 유지되지만 단어 선택에서 차이가 발생할 수 있습니다.
2. **중간.** `fasttext lid.176` 또는 `langdetect`를 사용하여 번역 출력에 언어 식별 검사를 구현하세요. MT 호출에 통합하여 목표 언어와 일치하지 않는 생성 결과가 반환되기 전에 감지되도록 하세요.
3. **어려움.** 선택한 도메인 코퍼스 5,000쌍으로 `nllb-200-distilled-600M`을 파인튜닝(fine-tuning)하세요. 파인튜닝 전후에 홀드아웃(held-out) 세트에서 BLEU 점수를 측정하고 어떤 유형의 문장이 개선되었고 어떤 문장이 성능이 저하되었는지 보고하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| BLEU | 번역 점수 | 간결성 패널티가 적용된 N-그램 정밀도. [0, 100]. |
| chrF | 문자 F-점수 | 문자 수준 F-점수. 형태학적으로 풍부한 언어에 더 민감함. |
| NMT | 신경망 기계 번역 | 병렬 텍스트로 학습된 트랜스포머 인코더-디코더. 2017년 이후 기본 방식. |
| NLLB | No Language Left Behind | 메타의 200개 언어 기계 번역 모델 패밀리. |
| 제약 디코딩 | 제어된 출력 | 출력에 특정 토큰 또는 N-그램이 나타나도록/나타나지 않도록 강제. |
| 환각 | 발명된 내용 | 소스에서 지원되지 않는 모델 출력. |

## 추가 자료

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 논문.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — `sacrebleu`가 BLEU 점수를 보고하는 유일한 올바른 방법인 이유.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 논문.
- [Hugging Face MT 가이드](https://huggingface.co/docs/transformers/tasks/translation) — 실용적인 파인튜닝(fine-tuning) 워크스루.