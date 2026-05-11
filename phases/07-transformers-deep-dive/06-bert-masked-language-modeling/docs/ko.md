# BERT — 마스크드 언어 모델링(Masked Language Modeling)

> GPT는 다음 단어를 예측합니다. BERT는 누락된 단어를 예측합니다. 한 문장의 차이 — 그리고 반 십년 동안 임베딩 형태의 모든 것.

**유형:** 구축(Build)
**언어:** Python
**사전 요구 사항:** Phase 7 · 05 (풀 트랜스포머), Phase 5 · 02 (텍스트 표현)
**소요 시간:** ~45분

## 문제

2018년에는 모든 NLP 작업(감정 분석, 개체명 인식, 질의응답, 함의 분석 등)이 자체 레이블 데이터로 처음부터 모델을 학습시켰습니다. 미세 조정할 수 있는 사전 학습된 "영어 이해" 체크포인트가 없었습니다. ELMo(2018)는 양방향 LSTM으로 문맥 임베딩을 사전 학습할 수 있음을 보여주었지만, 일반화에는 한계가 있었습니다.

BERT(Devlin et al. 2018)는 다음과 같은 질문을 던졌습니다: 트랜스포머 인코더를 가져와 인터넷상의 모든 문장으로 학습시키고, 양쪽 문맥에서 누락된 단어를 예측하도록 강제하면 어떨까? 그런 다음 다운스트림 작업에 하나의 헤드만 미세 조정합니다. 파라미터 효율성은 혁신적인 발견이었습니다.

결과: 18개월 내에 BERT와 그 변형 모델들(RoBERTa, ALBERT, ELECTRA)이 존재하는 모든 NLP 리더보드를 장악했습니다. 2020년까지 지구상의 모든 검색 엔진, 콘텐츠 조정 파이프라인, 시맨틱 검색 시스템에는 BERT가 내장되었습니다.

2026년에도 인코더 전용 모델은 분류, 검색, 구조화된 추출 작업에 적합한 도구입니다. 디코더보다 토큰당 5–10배 빠르게 실행되며, 임베딩은 모든 현대 검색 시스템의 핵심 기반입니다. ModernBERT(2024년 12월)는 Flash Attention + RoPE + GeGLU를 통해 8K 컨텍스트까지 아키텍처를 확장했습니다.

## 개념

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### 훈련 신호

문장 `the quick brown fox jumps over the lazy dog`를 가져옵니다.

15%의 토큰을 무작위로 마스킹합니다:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

마스킹된 위치에서 원본 토큰을 예측하도록 모델을 훈련시킵니다. 인코더가 양방향이므로 위치 1의 `[MASK]` 예측 시 위치 2+의 `brown fox jumps`를 활용할 수 있습니다. 이는 GPT가 할 수 없는 것입니다.

### BERT 마스킹 규칙

예측을 위해 선택된 15%의 토큰 중:

- 80%는 `[MASK]`로 대체됩니다.
- 10%는 무작위 토큰으로 대체됩니다.
- 10%는 변경되지 않습니다.

왜 항상 `[MASK]`를 사용하지 않을까요? 추론 시에는 `[MASK]`가 절대 나타나지 않기 때문입니다. 마스킹된 위치의 100%에 `[MASK]`를 기대하도록 훈련하면 사전 훈련과 파인튜닝 사이에 분포 차이가 발생합니다. 10% 무작위 + 10% 변경 없음은 모델이 정직하게 학습하도록 합니다.

### 다음 문장 예측(NSP) — 그리고 왜 제거되었는가

원본 BERT는 NSP로도 훈련되었습니다: 두 문장 A와 B가 주어졌을 때 B가 A 다음에 오는지 예측합니다. RoBERTa(2019)는 NSP를 제거했을 때 성능이 향상됨을 보였고, 현대 인코더는 이를 생략합니다.

### 2026년에 바뀐 것: ModernBERT

2024년 ModernBERT 논문은 2026년 기본 요소로 블록을 재구성했습니다:

| 구성 요소 | 원본 BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| 위치 인코딩 | 학습된 절대 위치 | RoPE |
| 활성화 함수 | GELU | GeGLU |
| 정규화 | LayerNorm | Pre-norm RMSNorm |
| 어텐션 | 전체 밀집 | 로컬(128) + 글로벌 교차 |
| 컨텍스트 길이 | 512 | 8192 |
| 토크나이저 | WordPiece | BPE |

2018년 스택과 달리 Flash-Attention을 네이티브로 지원합니다. 시퀀스 길이 8K에서 추론 속도가 DeBERTa-v3보다 2–3배 빠르며 GLUE 점수도 더 좋습니다.

### 2026년에도 인코더를 선택하는 사용 사례

| 작업 | 인코더가 디코더보다 나은 이유 |
|------|---------------------------|
| 검색 / 의미론적 검색 임베딩 | 양방향 컨텍스트 = 토큰당 더 나은 임베딩 품질 |
| 분류 (감정, 의도, 유해성) | 단일 순방향 전달; 생성 오버헤드 없음 |
| NER / 토큰 라벨링 | 위치별 출력, 네이티브 양방향 |
| 제로샷 함의 (NLI) | 인코더 상단에 분류기 헤드 |
| RAG를 위한 재랭커 | 교차-인코더 점수화, LLM 재랭커보다 10배 빠름 |

## 구축 방법

### 1단계: 마스킹 로직

`code/main.py`를 참조하세요. `create_mlm_batch` 함수는 토큰 ID 목록, 어휘 크기, 마스킹 확률을 입력으로 받습니다. 마스킹이 적용된 입력 ID와 레이블을 반환합니다(마스킹된 위치에만 레이블이 있고, 나머지 위치에는 -100 — PyTorch의 무시 인덱스 규칙).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: 원래 토큰 유지
    return input_ids, labels
```

### 2단계: 소규모 코퍼스에서 MLM 예측 실행

20개 단어로 구성된 어휘와 200개 문장으로 이루어진 데이터셋에 대해 2층 인코더 + MLM 헤드를 학습시킵니다. 그래디언트는 사용하지 않으며, 순전파 검증만 수행합니다. 전체 학습을 위해서는 PyTorch가 필요합니다.

### 3단계: 마스킹 유형 비교

3가지 규칙(80% 마스킹, 10% 랜덤 토큰, 10% 원본 유지)이 모델이 `[MASK]` 없이도 사용 가능하도록 유지하는 방식을 보여줍니다. 마스킹되지 않은 문장과 마스킹된 문장에 대해 예측을 수행합니다. 두 경우 모두 모델이 학습 중에 두 패턴을 모두 보았기 때문에 합리적인 토큰 분포를 생성해야 합니다.

### 4단계: 헤드 파인튜닝

장난감 감성 데이터셋에서 MLM 헤드를 분류 헤드로 교체합니다. 헤드만 학습시키고 인코더는 고정합니다. 이는 모든 BERT 응용 프로그램에서 따르는 패턴입니다.

## 사용 방법

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**임베딩 모델은 파인튜닝된 BERT입니다.** `sentence-transformers` 모델(예: `all-MiniLM-L6-v2`)은 대조 손실(contrastive loss)로 학습된 BERT입니다. 인코더(encoder)는 동일합니다. 손실 함수(loss function)가 변경되었습니다.

**크로스-인코더 재랭커(cross-encoder reranker) 역시 파인튜닝된 BERT입니다.** `[CLS] 쿼리 [SEP] 문서 [SEP]` 형식의 쌍 분류(pair-classification) 작업입니다. 쿼리와 문서 간의 양방향 어텐션(bidirectional attention)이 크로스-인코더가 바이인코더(biencoder) 대비 품질 우위를 가지는 핵심 요소입니다.

**2026년에 BERT를 선택하지 말아야 할 경우.** 생성(generative) 작업. 인코더는 토큰을 자기회귀적(autoregressive)으로 생성하는 합리적인 방법이 없습니다. 또한: 1B 파라미터 미만 규모에서 더 유연한 소형 디코더(decoder)가 품질을 맞출 수 있는 경우(Phi-3-Mini, Qwen2-1.5B).

## Ship It

`outputs/skill-bert-finetuner.md`를 참조하세요. 이 스킬은 새로운 분류 또는 추출 작업을 위해 BERT 파인튜닝(백본 선택, 헤드 사양, 데이터, 평가, 중단 조건)을 범위화합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하고 10,000개 토큰에 대한 마스크 분포를 출력하세요. 약 15%가 선택되고, 그 중 약 80%가 `[MASK]`가 되는지 확인하세요.
2. **중간.** 전체 단어 마스킹 구현: 단어가 서브워드로 토큰화될 경우, 모든 서브워드를 함께 마스킹하거나 전혀 마스킹하지 않도록 합니다. 500개 문장 코퍼스에서 이 방법이 MLM 정확도를 향상시키는지 측정하세요.
3. **어려움.** 공개 데이터셋의 10,000개 문장으로 2층, d=64 크기의 소형 BERT를 학습시키세요. SST-2 감정 분석을 위해 `[CLS]` 토큰을 파인튜닝(fine-tuning)하세요. 동일한 파라미터 규모의 디코더 전용(decoder-only) 베이스라인과 비교 — 어떤 모델이 더 좋은 성능을 내는지 확인하세요?

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|----------|
| MLM | "마스킹된 언어 모델링" | 훈련 신호: 토큰의 15%를 무작위로 `[MASK]`로 대체, 원본 예측. |
| Bidirectional | "양방향 보기" | 인코더 어텐션에 인과적 마스크 없음 — 모든 위치가 다른 모든 위치를 참조. |
| `[CLS]` | "풀러 토큰" | 모든 시퀀스 앞에 추가되는 특수 토큰; 최종 임베딩이 문장 수준 표현으로 사용됨. |
| `[SEP]` | "세그먼트 구분자" | 쌍으로 된 시퀀스(예: 쿼리/문서, 문장 A/B)를 구분. |
| NSP | "다음 문장 예측" | BERT의 두 번째 사전 훈련 작업; RoBERTa에서 쓸모없음이 확인되어 2019년 이후 제외. |
| Fine-tuning | "작업에 적응" | 인코더를 대부분 고정; 다운스트림 작업을 위해 상단에 작은 헤드 훈련. |
| Cross-encoder | "재순위 결정기" | 쿼리와 문서를 모두 입력으로 받아 관련성 점수를 출력하는 BERT. |
| ModernBERT | "2024년 리프레시" | RoPE, RMSNorm, GeGLU, 교번적 지역/전역 어텐션, 8K 컨텍스트로 재구축된 인코더.

## 추가 자료

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 원본 논문.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — BERT를 올바르게 학습시키는 방법; NSP 제거.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 대체 토큰 감지가 동일한 계산량에서 MLM을 능가.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 논문.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 표준 인코더 구현 참조.