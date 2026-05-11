# GPT — 인과적 언어 모델링

> BERT는 양쪽을 모두 본다. GPT는 과거만 본다. 삼각형 마스킹은 현대 AI에서 가장 중요한 단일 코드 라인이다.

**유형:** 구축
**언어:** Python
**사전 요구 사항:** 7단계 · 02 (셀프 어텐션), 7단계 · 05 (전체 트랜스포머), 7단계 · 06 (BERT)
**소요 시간:** ~75분

## 문제 정의

언어 모델은 하나의 질문에 답한다: 처음 `t-1`개의 토큰이 주어졌을 때, 토큰 `t`에 대한 확률 분포는 무엇인가? 이 신호 — 다음 토큰 예측 — 로 학습하면 한 번에 하나의 토큰씩 임의의 텍스트를 생성할 수 있는 모델이 만들어진다.

전체 시퀀스를 병렬로 종단간(end-to-end) 학습하려면 각 위치의 예측이 오직 이전 위치들만을 의존해야 한다. 그렇지 않으면 모델이 정답을 보고 치트하는 것이 가능해진다.

인과적 마스킹(causal mask)이 이 역할을 한다. 이는 소프트맥스(softmax) 전에 어텐션 스코어에 더해지는 `-inf` 값으로 구성된 단일 상삼각 행렬(upper-triangular matrix)이다. 소프트맥스 적용 후 해당 위치들은 0이 된다. 각 위치는 자신과 이전 위치들만을 어텐션할 수 있다. 그리고 전체 시퀀스에 한 번 적용되기 때문에 단일 순전파(forward pass)에서 N개의 병렬 다음 토큰 예측이 가능해진다.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2024), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — 이들은 모두 동일한 핵심 루프(core loop)를 가진 디코더 전용 인과적 트랜스포머다. 단지 더 큰 규모, 더 나은 데이터, 더 나은 RLHF(강화 학습 기반 인간 피드백)로 진화했을 뿐이다.

## 개념

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### 마스크

길이 `N`의 시퀀스가 주어졌을 때, `N × N` 행렬을 구성합니다:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

소프트맥스 전에 원시 어텐션 점수에 `M`을 더합니다. `exp(-inf) = 0`이므로 마스크된 위치는 가중치가 0이 됩니다. 어텐션 행렬의 각 행은 이전 위치들에 대한 확률 분포입니다.

구현 비용: `torch.tril()` 호출 1회. 계산 시간: 나노초. 영향력: 모든 것.

### 병렬 훈련, 직렬 추론

훈련: `(N, d_model)` 시퀀스 전체를 한 번 순전파하고, N개의 크로스엔트로피 손실(각 위치당 1개)을 계산한 후 합산하고 역전파합니다. 시퀀스 방향으로 병렬화됩니다. 이것이 GPT 훈련이 확장 가능한 이유입니다 — 배치 내 1M 토큰을 한 번의 GPU 패스로 처리합니다.

추론: 토큰을 하나씩 생성합니다. `[t1, t2, t3]`를 입력으로 받아 `t4`를 얻습니다. `[t1, t2, t3, t4]`를 입력으로 받아 `t5`를 얻습니다. `[t1, t2, t3, t4, t5]`를 입력으로 받아 `t6`를 얻습니다. KV 캐시(레슨 12)는 `t1…tn`의 은닉 상태를 저장하여 매 단계마다 재계산하지 않도록 합니다. 하지만 추론 시 직렬 깊이는 출력 길이와 같습니다. 이것이 자기회귀적 비용이며, 디코딩이 모든 LLM의 지연 시간 병목인 이유입니다.

### 손실 — 시프트-바이-원

토큰 `[t1, t2, t3, t4]`가 주어졌을 때:

- 입력: `[t1, t2, t3]`
- 타겟: `[t2, t3, t4]`

모든 위치 `i`에 대해 `-log P(target_i | inputs[:i+1])`를 계산합니다. 이를 합산하면 전체 시퀀스의 크로스엔트로피가 됩니다.

들어본 모든 트랜스포머 언어 모델은 이 손실로 훈련됩니다. 사전 훈련, 파인튜닝, SFT — 동일한 손실, 다른 데이터.

### 디코딩 전략

훈련 후, 샘플링 선택은 사람들이 생각하는 것보다 더 중요합니다.

| 방법 | 동작 | 사용 시기 |
|--------|--------------|-------------|
| 그리디 | 매 단계 최대값 선택 | 결정적 작업, 코드 완성 |
| 온도 | 로짓을 T로 나누고 샘플링 | 창의적 작업, 높은 T = 더 많은 다양성 |
| Top-k | 상위 k 토큰에서만 샘플링 | 낮은 확률 꼬리 제거 |
| Top-p (nucleus) | 누적 확률이 p 이상인 최소 집합에서 샘플링 | 2020+ 기본값; 분포 형태에 적응 |
| Min-p | `p > min_p * max_p`인 토큰 유지 | 2024+; Top-p보다 긴 꼬리 거부에 더 효과적 |
| 추측 디코딩 | 초안 모델이 N 토큰 제안, 큰 모델이 검증 | 동일 품질에서 2–3배 지연 시간 감소 |

2026년에는 min-p + 온도 0.7이 오픈웨이트 모델의 합리적인 기본값입니다. 추측 디코딩은 모든 프로덕션 추론 스택의 기본 요건입니다.

### "GPT 레시피"의 성공 요인

1. **디코더 전용.** 인코더 오버헤드 없음. 각 레이어당 어텐션 + FFN 1회 순전파.
2. **확장성.** 124M → 1.5B → 175B → 수조. Chinchilla 확장 법칙(레슨 13)은 컴퓨트 사용 방법을 알려줍니다.
3. **컨텍스트 학습.** 6B–13B에서 등장. 모델은 파인튜닝 없이 몇 가지 예시를 따를 수 있습니다.
4. **RLHF.** 인간 선호도에 대한 사후 훈련은 원시 사전 훈련 텍스트를 채팅 도우미로 변환했습니다.
5. **사전 정규화 + RoPE + SwiGLU.** 대규모에서 안정적인 훈련.

핵심 아키텍처는 GPT-2 이후로 크게 변하지 않았습니다. 모든 흥미로운 발전은 데이터, 규모, 사후 훈련에서 일어났습니다.

## 구축 방법

### 단계 1: 인과적 마스킹

`code/main.py`를 참조하세요. 한 줄짜리 코드:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

소프트맥스(softmax) 이전에 어텐션 점수(attention scores)에 추가합니다. 이것이 전체 메커니즘입니다.

### 단계 2: 2층 GPT-ish 모델

두 개의 디코더 블록(마스킹된 자기 어텐션 + 피드포워드 네트워크, 교차 어텐션 없음)을 쌓습니다. 토큰 임베딩(token embedding), 위치 인코딩(positional encoding), 언임베딩(unembedding)을 추가합니다(토큰 임베딩 행렬과 연결 — GPT-2 이후 표준 기법).

### 단계 3: 다음 토큰 예측, 엔드-투-엔드

20토큰 장난감 어휘에서 모든 위치에 로짓(logits)을 생성합니다. 시프트-바이-원(shift-by-one) 타겟에 대해 교차 엔트로피 손실(cross-entropy loss)을 계산합니다. 그래디언트는 사용하지 않습니다 — 순전파(forward-pass) 검증입니다.

### 단계 4: 샘플링

탐욕적(greedy), 온도(temperature), 탑-k(top-k), 탑-p(top-p), 최소 확률(min-p) 샘플링을 구현합니다. 고정된 프롬프트에 대해 각각 실행하고 출력을 비교합니다. 샘플링 함수는 10줄 이내로 작성 가능합니다.

## 사용 방법

PyTorch, 2026년 관용구:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

`generate()` 내부에서는 순전파(forward pass)를 실행하고, 최종 위치 로짓(logits)을 추출한 후 다음 토큰을 샘플링하고, 이를 추가하며 반복합니다. 모든 프로덕션 LLM 추론 스택(vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX)은 배치 전처리(batched prefill), 연속 배치(continuous batching), KV 캐시 페이징(KV cache paging), 추측 디코딩(speculative decoding) 등의 강력한 최적화를 적용하여 동일한 루프를 구현합니다.

**GPT vs BERT, 한 줄로 설명:**  
GPT는 `P(x_t | x_{<t})`를 예측합니다. BERT는 `P(x_masked | x_unmasked)`를 예측합니다. 손실 함수(loss function)가 모델의 생성 가능 여부를 결정합니다.

## Ship It

`outputs/skill-sampling-tuner.md`를 참조하세요. 이 스킬은 새로운 생성 작업을 위한 샘플링 파라미터를 선택하고 결정적 디코딩이 필요한 경우 플래그를 지정합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하고 소프트맥스 후 인과적 어텐션 행렬이 하삼각행렬인지 확인하세요. 스팟 체크: 3행에는 0–3열에만 가중치가 있어야 합니다.
2. **중간.** 너비 4의 빔 서치를 구현하세요. 10개의 짧은 프롬프트에서 빔-4와 그리디 방식의 퍼플렉서티를 비교하세요. 빔이 항상 더 나은가요? (힌트: 일반적으로 번역에서는 그렇지만 개방형 채팅에서는 아닐 수 있습니다.)
3. **어려움.** 추측적 디코딩을 구현하세요: 2층 소형 모델을 드래프트로, 6층 모델을 검증기로 사용하세요. 길이 64의 100개 완성 작업에서 벽시계 속도 향상을 측정하세요. 검증기의 그리디 결과와 출력이 일치하는지 확인하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| 인과적 마스킹(Causal mask) | "삼각형" | 어텐션 점수에 추가되는 상삼각 `-inf` 행렬로, 위치 `i`는 `≤ i` 위치만 볼 수 있음. |
| 다음 토큰 예측(Next-token prediction) | "손실" | 모든 위치에서 모델의 분포와 실제 다음 토큰 간의 교차 엔트로피. |
| 자기회귀적(Autoregressive) | "한 번에 하나씩 생성" | 출력을 입력으로 다시 사용; 병렬 처리는 학습 시에만 가능, 생성 시에는 불가능. |
| 로짓(Logits) | "소프트맥스 이전 점수" | 소프트맥스 적용 전 LM 헤드의 원시 출력; 샘플링은 이 값에서 발생. |
| 온도(Temperature) | "창의성 조절기" | 로짓을 T로 나눔; T→0 = 탐욕적, T→∞ = 균일 분포. |
| Top-p | "뉴클리어스 샘플링" | 누적 확률이 ≥p가 되는 최소 집합으로 분포 자름; 남은 부분에서 샘플링. |
| Min-p | "Top-p보다 나은 방법" | `p ≥ min_p × max_p`인 토큰 유지; 분포 날카로움에 따라 커트오프 조정. |
| 추측적 디코딩(Speculative decoding) | "초안 + 검증" | 저렴한 모델이 N개 토큰 제안; 큰 모델이 병렬로 검증. |
| 티처 포싱(Teacher forcing) | "학습 트릭" | 학습 시 모델 예측이 아닌 실제 이전 토큰을 입력으로 사용. 모든 seq2seq LM의 표준 방법.

## 추가 자료

- [Radford et al. (2018). 생성적 사전 학습을 통한 언어 이해 개선](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). 언어 모델은 비지도 다중 작업 학습자](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). 언어 모델은 소량 학습기](https://arxiv.org/abs/2005.14165) — GPT-3 및 인-컨텍스트 학습.
- [Leviathan, Kalman, Matias (2023). 추측 디코딩을 통한 트랜스포머의 빠른 추론](https://arxiv.org/abs/2211.17192) — 스펙 디코딩 논문.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 표준 인과적 언어 모델 참조 코드.