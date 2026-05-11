# 전체 트랜스포머 — 인코더 + 디코더

> 어텐션(attention)이 주인공이다. 나머지 — 잔차(residuals), 정규화(normalization), 피드포워드(feed-forward), 크로스 어텐션(cross-attention) — 는 이를 깊게 쌓을 수 있게 하는 비계(scaffolding)다.

**유형:** 구축(Build)  
**언어:** Python  
**선수 지식:** 7단계 · 02 (셀프 어텐션), 7단계 · 03 (멀티 헤드 어텐션), 7단계 · 04 (위치 인코딩)  
**소요 시간:** ~75분

## 문제 정의

단일 어텐션(attention) 레이어는 특징 추출기(feature extractor)일 뿐 모델이 아니다. 레이어당 하나의 행렬 곱셈(matmul)만으로는 언어 처리에 충분한 용량(capacity)을 제공하지 못한다. 깊이(depth)가 필요하며, 적절한 연결 구조(plumbing) 없이는 깊이를 확보할 수 없다.

2017년 Vaswani 논문은 하나의 어텐션 레이어를 쌓아 올릴 수 있는 블록(block)으로 변환한 6가지 설계 결정을 제시했다. 이후 모든 트랜스포머(encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5))는 동일한 기본 구조(skeleton)를 계승했다. 2026년에는 블록들이 개선되었지만(RMSNorm, SwiGLU, pre-norm, RoPE), 기본 구조는 동일하다.

이 레슨은 바로 그 기본 구조를 다룬다. 다음 레슨에서는 이를 특수화한다 — 06은 인코더(encoder), 07은 디코더(decoder), 08은 인코더-디코더(encoder-decoder)에 대해 설명한다.

## 개념

![인코더와 디코더 블록 내부, 연결 구조](../assets/full-transformer.svg)

### 6가지 구성 요소

1. **임베딩(embedding) + 위치 신호(positional signal).** 토큰 → 벡터. 위치는 RoPE(현대적) 또는 사인파(고전적)를 통해 주입.
2. **셀프 어텐션(self-attention).** 모든 위치가 다른 모든 위치를 참조. 디코더에서는 마스킹 적용.
3. **피드포워드 네트워크(FFN).** 위치별 2층 MLP: `W_2 · activation(W_1 · x)`. 기본 확장 비율 4×.
4. **잔차 연결(residual connection).** `x + sublayer(x)`. 이 연결이 없으면 ~6층 이후 그래디언트 소실 발생.
5. **레이어 정규화(layer normalization).** `LayerNorm` 또는 `RMSNorm`(현대적). 잔차 스트림 안정화.
6. **크로스 어텐션(cross-attention, 디코더 전용).** 쿼리는 디코더에서, 키와 값은 인코더 출력에서 가져옴.

### 인코더 블록 (BERT, T5 인코더에서 사용)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── 잔차 연결 ──┘
```

인코더는 양방향. 마스킹 없음. 모든 위치가 다른 모든 위치를 참조.

### 디코더 블록 (GPT, T5 디코더에서 사용)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

디코더는 블록당 3개의 서브레이어. 중간 서브레이어인 크로스 어텐션이 인코더에서 디코더로 정보가 흐르는 유일한 경로. 순수 디코더 전용 아키텍처(GPT)에서는 크로스 어텐션이 생략되고 마스킹된 셀프 어텐션 + FFN만 사용.

### 사전 정규화(pre-norm) vs 사후 정규화(post-norm)

원본 논문: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. 사후 정규화는 2019년경부터 선호되지 않음 — 신중한 워밍업 없이는 깊은 모델 학습이 어려움. 사전 정규화(`LN` *서브레이어 전*)가 2026년 기본 방식: Llama, Qwen, GPT-3+, Mistral 모두 사용.

### 2026년 현대화된 블록

Vaswani 2017은 LayerNorm + ReLU를 사용. 현대 스택은 둘 다 대체. 실제 프로덕션 블록의 구성:

| 구성 요소 | 2017 | 2026 |
|-----------|------|------|
| 정규화 | LayerNorm | RMSNorm |
| FFN 활성화 함수 | ReLU | SwiGLU |
| FFN 확장 비율 | 4× | 2.6× (SwiGLU는 3개의 행렬 사용, 총 파라미터 수 동일) |
| 위치 인코딩 | 사인파 절대 위치 | RoPE |
| 어텐션 | 전체 MHA | GQA(또는 MLA) |
| 바이어스 항 | 있음 | 없음 |

RMSNorm은 LayerNorm의 평균 중심화(mean-centering)를 제거(뺄셈 1회 감소)하여 계산량을 절약하고 경험적으로 안정적. SwiGLU(`Swish(W1 x) ⊙ W3 x`)는 Llama, PaLM, Qwen 논문에서 ReLU/GELU FFN보다 ~0.5점 ppl 성능 향상.

### 파라미터 수

`d_model = d` 및 FFN 확장 비율 `r`인 블록 1개 기준:

- MHA: `4 · d²` (Q, K, V, O 투영)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- 정규화: 무시할 수준

`d = 4096, r = 2.6, layers = 32`(대략 Llama 3 8B) 시 총 파라미터:  
`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 파라미터/층 × 32 ≈ 7B`(임베딩 및 헤드 제외). 공개된 수치와 일치.

## 구축 방법

### 1단계: 구성 요소

Lesson 03의 독립적인 `Matrix` 클래스를 사용:

- `layer_norm(x, eps=1e-5)` — 평균 빼기, 표준편차로 나누기.
- `rms_norm(x, eps=1e-6)` — RMS로 나누기. 평균 빼기 없음.
- `gelu(x)` 및 `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` 및 `decoder_block(x, enc_out, params)`.

전체 배선은 `code/main.py` 참조.

### 2단계: 2층 인코더와 2층 디코더 연결

쌓아 올리기. 인코더 출력을 모든 디코더 크로스 어텐션에 전달. 출력 프로젝션 전에 최종 LN 추가.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 3단계: 장난감 예제에서 순전파 실행

6토큰 소스와 5토큰 타겟을 입력. 출력 형태가 `(5, vocab)`인지 확인. 훈련 없음 — 이 레슨은 아키텍처에 관한 내용이며 손실 함수와 무관.

### 4단계: RMSNorm + SwiGLU로 교체

LayerNorm과 ReLU-FFN을 RMSNorm과 SwiGLU로 대체. 형태가 여전히 일치하는지 확인. 이는 단일 함수 교체로 2026년 현대화를 구현한 것.

## 사용 방법

PyTorch/TF 참조 구현: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. 하지만 대부분의 2026년 프로덕션 코드는 자체 블록을 구현합니다. 이유는 다음과 같습니다:

- Flash Attention은 `nn.MultiheadAttention`이 아닌 어텐션 내부에서 호출됩니다.
- GQA / MLA는 표준 라이브러리 참조에 없습니다.
- RoPE, RMSNorm, SwiGLU는 PyTorch 기본값이 아닙니다.

HF `transformers`에는 참고해야 할 깔끔한 참조 블록이 있습니다: `modeling_llama.py`는 2026년 디코더 전용 블록의 표준입니다. 약 500줄이며 한 번 살펴볼 가치가 있습니다.

**인코더 vs 디코더 vs 인코더-디코더 — 선택 기준:**

| 필요 사항 | 선택 | 예시 |
|------|------|---------|
| 분류, 임베딩, 텍스트 기반 QA | 인코더 전용 | BERT, DeBERTa, ModernBERT |
| 텍스트 생성, 채팅, 코드, 추론 | 디코더 전용 | GPT, Llama, Claude, Qwen |
| 구조화된 입력 → 구조화된 출력 (번역, 요약) | 인코더-디코더 | T5, BART, Whisper |

디코더 전용이 언어를 지배한 이유는 가장 깔끔하게 확장되며 이해(comprehension)와 생성(generation)을 모두 처리하기 때문입니다. 인코더-디코더는 입력에 명확한 "소스 시퀀스" 정체성이 있는 경우(번역, 음성 인식, 구조화된 작업) 여전히 최적입니다.

## Ship It

`outputs/skill-transformer-block-reviewer.md`를 참조하세요. 이 스킬은 새로운 트랜스포머 블록 구현을 2026년 기본값과 비교하여 누락된 부분(사전 정규화(pre-norm), RoPE(Rotary Position Embedding), RMSNorm, GQA(Grouped Query Attention), FFN 확장 비율)을 식별합니다.

## 연습 문제

1. **쉬움.** `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`에서 인코더 블록(encoder_block)의 파라미터 수를 계산하세요. 블록을 구현하고 `sum(p.numel() for p in block.parameters())`를 사용하여 검증하세요.
2. **중간.** 포스트-노름(post-norm)에서 프리-노름(pre-norm)으로 전환하세요. 둘 다 초기화하고 무작위 입력에 대해 12개 쌓인 레이어 후 활성화 노름(activation norm)을 측정하세요. 포스트-노름의 활성화는 발산해야 하며, 프리-노름의 활성화는 유계(bounded) 상태를 유지해야 합니다.
3. **어려움.** 장난감 복사 작업(입력 `x`를 역순으로 복사)에 4층 인코더-디코더(encoder-decoder)를 구현하세요. 100단계 훈련 후 손실(loss)을 보고하세요. RMSNorm + SwiGLU + RoPE로 교체할 때 손실이 감소하는지 확인하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 블록(Block) | "하나의 트랜스포머 레이어" | 정규화(norm) + 어텐션(attention) + 정규화(norm) + 피드포워드 네트워크(FFN)의 스택. 잔차 연결(residual connection)로 감싸져 있음. |
| 잔차(Residual) | "스킵 연결(skip connection)" | `x + f(x)` 출력; 깊은 스택을 통한 그래디언트 흐름(gradient flow) 가능. |
| 프리-노름(Pre-norm) | "뒤가 아닌 앞에 정규화" | 현대적 접근: `x + sublayer(LN(x))`. 워밍업(warmup) 없이도 더 깊은 모델 학습 가능. |
| RMSNorm | "평균을 제외한 레이어노름(LayerNorm)" | RMS로 나눔; 연산 1회 감소, 경험적 안정성은 동일. |
| SwiGLU | "모두가 전환한 FFN" | `Swish(W1 x) ⊙ W3 x → W2`. LM 퍼플렉서티(perplexity)에서 ReLU/GELU를 능가. |
| 크로스 어텐션(Cross-attention) | "디코더가 인코더를 보는 방식" | 디코더의 Q(Query)와 인코더 출력의 K(Key)/V(Value)를 사용하는 다중 헤드 어텐션(MHA). |
| FFN 확장(FFN expansion) | "중간 MLP의 너비" | 은닉 크기(hidden-size) 대 d_model 비율, 일반적으로 4(레이어노름) 또는 2.6(SwiGLU). |
| 바이어스 프리(Bias-free) | "+b 항 제거" | 현대 스택은 선형 레이어의 바이어스 생략; 약간의 퍼플렉서티 개선, 모델 크기 감소.

## 추가 자료

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 원본 블록 사양.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 사전 정규화(pre-norm)가 사후 정규화(post-norm)보다 깊은 모델에서 우수한 이유.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 논문.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026년 기준 디코더 전용 블록의 표준 구현.