# 처음부터 트랜스포머 구축하기 — 캡스톤 프로젝트

> 13개의 레슨. 하나의 모델. 단축키 없음.

**유형:** 구축
**언어:** Python
**선수 조건:** Phase 7 · 01부터 13까지. 건너뛰다간 후회함.
**소요 시간:** ~120분

## 문제

모든 논문을 읽었고, 어텐션(attention), 멀티헤드 분할(multi-head splits), 위치 인코딩(positional encodings), 인코더 및 디코더 블록(encoder and decoder blocks), BERT 및 GPT 손실 함수(BERT and GPT losses), MoE(Mixture of Experts), KV 캐시(KV cache)를 구현했습니다. 이제 이들을 실제 작업에서 함께 작동하게 만들어야 합니다.

최종 프로젝트: 문자 수준 언어 모델링 작업(character-level language modeling task)에 대해 작은 디코더 전용 트랜스포머(decoder-only transformer)를 종단간(end-to-end)으로 학습시킵니다. 이 모델은 셰익스피어 작품을 읽고, 새로운 셰익스피어 작품을 생성하며, 10분 이내에 노트북에서 학습될 수 있을 만큼 작습니다. 또한 더 큰 데이터셋과 긴 학습 시간으로 교체하면 실제 언어 모델(LM)을 얻을 수 있을 정도로 정확합니다.

이것은 강의의 "나노GPT(nanoGPT)"입니다. 이 모델은 독창적이지 않습니다 — Karpathy의 2023년 나노GPT 튜토리얼이 모든 학생이 한 번 이상 작성하는 참조 구현체입니다. 우리는 이 구조를 가져와 강의에서 다룬 내용을 중심으로 재설계합니다.

## 개념

![Transformer-from-scratch 블록 다이어그램](../assets/capstone.svg)

주석이 달린 아키텍처:

```
입력 토큰 (B, N)
   │
   ▼
토큰 임베딩 + 위치 임베딩  ◀── 레슨 04 (RoPE 옵션)
   │
   ▼
┌──── 블록 × L ────────────────────┐
│  RMSNorm                          │  ◀── 레슨 05
│  MultiHeadAttention (인과적)      │  ◀── 레슨 03 + 07 (인과적 마스크)
│  잔차 연결                        │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── 레슨 05
│  잔차 연결                        │
└────────────────────────────────── ┘
   │
   ▼
최종 RMSNorm
   │
   ▼
lm_head (토큰 임베딩과 공유됨)
   │
   ▼
로그잇 (B, N, V)
   │
   ▼
1-스텝 시프트 교차 엔트로피            ◀── 레슨 07
```

### 제공하는 내용

- `GPTConfig` — 모든 하이퍼파라미터를 구성하는 단일 위치.
- `MultiHeadAttention` — 인과적, 배치 처리, 선택적 Flash 스타일 경로 (PyTorch의 `scaled_dot_product_attention`).
- `SwiGLUFFN` — 현대적 FFN.
- `Block` — 사전 정규화, 잔차 연결로 감싸진 어텐션 + FFN.
- `GPT` — 임베딩, 쌓인 블록, LM 헤드, generate().
- AdamW, 코사인 학습률, 그래디언트 클리핑을 포함한 훈련 루프.
- 셰익스피어 텍스트에 대한 문자 수준 토크나이저.

### 제공하지 않는 내용

- RoPE — 레슨 04에서 개념적으로 구현됨. 여기서는 단순성을 위해 학습된 위치 임베딩을 사용. 연습 문제에서는 RoPE로 교체하도록 요청.
- 생성 중 KV 캐시 — 각 생성 단계에서 전체 접두사에 대한 어텐션을 재계산. 느리지만 단순. 연습 문제에서는 KV 캐시 추가를 요청.
- Flash Attention — PyTorch 2.0+는 입력이 일치할 경우 자동 디스패치; `F.scaled_dot_product_attention` 사용.
- MoE — 블록당 단일 FFN. 레슨 11에서 MoE를 확인.

### 목표 지표

Mac M2 노트북에서 4층, 4헤드, d_model=128 GPT가 `tinyshakespeare.txt`에 대해 2,000단계 훈련 시:

- 훈련 손실은 ~4.2(무작위)에서 약 6분 만에 ~1.5로 수렴.
- 샘플링된 출력은 셰익스피어 형태: 고어, 줄바꿈, "ROMEO:"와 같은 고유명사가 나타남.
- 검증 손실(텍스트의 마지막 10% 홀드아웃)은 훈련 손실과 밀접하게 추적; 이 크기/예산에서는 과적합 없음.

## 빌드하기

이 레슨은 PyTorch를 사용합니다. `torch`를 설치하세요(CPU 빌드도 괜찮습니다). `code/main.py`를 참조하세요. 스크립트는 다음을 처리합니다:

- `tinyshakespeare.txt`가 없을 경우 다운로드(또는 로컬 복사본 읽기).
- 바이트 수준 문자 토크나이저.
- 90/10 비율의 학습/검증 분할.
- 지원되는 하드웨어에서 bf16 자동 캐스팅을 사용한 학습 루프.
- 학습 완료 후 샘플링.

### 1단계: 데이터

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65개의 고유 문자. 작은 어휘 집합. 4바이트 `vocab_size`에 적합합니다. BPE도 없고 토크나이저 문제도 없습니다.

### 2단계: 모델

`code/main.py`를 참조하세요. 이 블록은 레슨 05의 교과서 내용입니다 — 사전 정규화(pre-norm), RMSNorm, SwiGLU, 인과적 MHA. 4/4/128 파라미터 수는 약 800K입니다.

### 3단계: 학습 루프

길이 256의 토큰 윈도우를 무작위로 배치로 가져옵니다. 순전파. 시프트-바이-원 크로스 엔트로피. 역전파. AdamW 단계. 로그 기록. 반복.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 4단계: 샘플링

프롬프트가 주어지면 반복적으로 순전파를 수행하고, 상위 p 로그이트에서 샘플링한 후 추가하고 계속합니다. 500개 토큰 후 중지합니다.

### 5단계: 출력 읽기

2,000단계 후:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

셰익스피어는 아닙니다. 하지만 셰익스피어 형태입니다. 약 800K 파라미터와 노트북에서 6분이라는 명확한 승리입니다.

## 사용 방법

이 캡스톤은 참조 아키텍처입니다. 실제 배포를 위한 세 가지 확장 방법:

1. **토크나이저 교체.** BPE(예: `tiktoken.get_encoding("cl100k_base")`) 사용. 어휘 크기가 65에서 ~50,000으로 증가. 모델 용량을 확장해야 함.
2. **더 큰 코퍼스 학습.** `OpenWebText` 또는 `fineweb-edu`(HuggingFace) 사용. 10B 토큰을 단일 A100에서 125M 파라미터 GPT로 학습 시 약 24시간 소요.
3. **RoPE + KV 캐시 + Flash Attention 추가.** 아래 연습에서 각각을 단계별로 설명합니다.

최종적으로 유창한 영어를 생성하는 125M 파라미터 GPT가 완성됩니다. 최전선 모델은 아니지만, 동일한 코드 경로(규모만 더 큰)로 Karpathy, EleutherAI, Allen Institute이 2026년 연구 체크포인트 학습에 사용하는 것과 동일합니다.

## Ship It

`outputs/skill-transformer-review.md`를 참조하세요. 이 기술 리뷰는 이전 13개 레슨 전반에 걸쳐 처음부터 구현한 트랜스포머(transformer)의 정확성을 검토합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행합니다. 학습된 모델의 최종 검증 손실(val loss)이 2.0 미만인지 확인합니다. `max_steps`를 2,000에서 5,000으로 변경합니다 — 검증 손실이 계속 개선되나요?
2. **중간.** 학습된 위치 임베딩(learned positional embeddings)을 RoPE(Rotary Position Embedding)로 교체합니다. `MultiHeadAttention` 내부에서 Q(Query)와 K(Key)에 회전을 적용합니다. 학습 후 검증 손실이 최소한 기존과 동일하게 낮은지 확인합니다.
3. **중간.** 샘플링 루프에 KV 캐시(KV cache)를 구현합니다. 캐시 사용 유무에 따라 500개의 토큰을 생성합니다. 노트북 환경에서 벽시계 시간(wall-clock time)이 5–20배 개선되어야 합니다.
4. **어려움.** 다음 토큰뿐만 아니라 "다음-플러스-원" 토큰(MTP — Multi-Token Prediction, DeepSeek-V3에서 제안)을 예측하는 두 번째 헤드를 모델에 추가합니다. 공동 학습(jointly training) 후 성능 향상 여부를 확인합니다.
5. **어려움.** 각 블록당 단일 피드포워드 네트워크(FFN)를 4개의 전문가(MoE: Mixture of Experts)로 교체합니다. 라우터(router) + 상위 2개 라우팅(top-2 routing)을 적용합니다. 활성 파라미터 수를 동일하게 유지했을 때 검증 손실 변화를 관찰합니다.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| nanoGPT | "Karpathy의 튜토리얼 레포지토리" | 최소 디코더 전용 트랜스포머 학습 코드, ~300 LOC; 표준 참조 구현. |
| tinyshakespeare | "표준 장난감 코퍼스" | ~1.1MB 텍스트; 2015년 이후 모든 문자 기반 언어 모델 튜토리얼에서 사용. |
| Tied 임베딩 | "입력/출력 행렬 공유" | LM 헤드 가중치 = 토큰 임베딩 행렬의 전치; 파라미터 절약 및 품질 향상. |
| bf16 autocast | "학습 정밀도 트릭" | 순전파/역전파를 bf16으로 실행, 옵티마이저 상태는 fp32 유지; 2021년 이후 표준. |
| 그래디언트 클리핑 | "스파이크 방지" | 전역 그래디언트 노름 1.0으로 제한; 학습 폭주 방지. |
| 코사인 학습률 스케줄 | "2020+ 기본값" | 학습률 선형 증가(워밍업) 후 코사인 형태로 최대치의 10%까지 감소. |
| MFU | "모델 FLOP 활용도" | 달성 FLOPs / 이론적 최대치; 2026년 기준 40%(밀집), 30%(MoE)면 우수. |
| 검증 손실 | "홀드아웃 손실" | 모델이 본 적 없는 데이터에 대한 교차 엔트로피; 과적합 감지기. |

## 추가 자료

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 클래식한 주석이 달린 구현체.