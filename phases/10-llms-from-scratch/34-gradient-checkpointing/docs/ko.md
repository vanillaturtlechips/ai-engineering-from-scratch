# 그래디언트 체크포인팅과 활성화 재계산

> 역전파는 모든 중간 활성화 값을 저장합니다. 70B 파라미터와 128K 컨텍스트에서 이는 랭크당 3TB의 활성화 값을 의미합니다. 체크포인팅은 FLOPs와 메모리를 교환합니다: 저장 대신 재계산합니다. 문제는 어떤 세그먼트를 버릴지이며, 그 답은 "모두"가 아닙니다.

**유형:** 구축(Build)
**언어:** Python (numpy, 선택적 torch 사용)
**선수 지식:** Phase 10 Lesson 04 (사전 학습 미니-GPT), Phase 10 Lesson 05 (확장 및 분산 처리)
**소요 시간:** ~70분

## 문제

트랜스포머를 훈련시킬 때, 각 레이어마다 역전파(backpropagation)에서 미분되는 모든 연산(op)의 입력값을 저장합니다. 여기에는 어텐션(attention) 입력, Q/K/V 프로젝션(projection), 소프트맥스(softmax) 출력, 피드포워드 네트워크(FFN) 입력, 정규화(norm) 출력, 그리고 잔차(residual) 스트림이 포함됩니다. 은닉 크기 `d`, 시퀀스 길이 `L`, 배치 `B`가 주어졌을 때, 이는 레이어당 약 `12 * B * L * d`개의 부동소수점(float) 수준입니다.

`d=8192, L=8192, B=1`인 경우, BF16 형식으로 레이어당 800MB가 필요합니다. 64레이어 모델은 51GB의 활성화(activations) 메모리가 소모되며, 이는 마이크로배치(microbatch) 크기를 곱하기 전, 어텐션-소프트맥스 중간값(헤드당 `L^2`)을 추가하기 전, 텐서 병렬(tensor-parallel) 부분 복사본을 고려하기 전의 수치입니다.

양면 부담: BF16 가중치와 옵티마이저(optimizer) 상태는 80GB에 들어갈 수 있지만, 활성화 메모리가 이를 초과합니다. 기울기 체크포인팅(gradient checkpointing, 활성화 재계산)이 표준 해결책입니다. 대부분의 활성화를 삭제한 후, 역전파 중에 순전파를 다시 수행하여 활성화 값을 복원합니다. 비용: 추가 FLOPs. 이점: 체크포인트 세그먼트와 총 레이어 수의 비율에 따라 메모리가 감소합니다.

순진하게 체크포인팅을 구현하면 단계당 약 33% 더 많은 순전파 FLOPs가 소모됩니다. Korthikanti 등의 "스마트 선택(smart selection)"에 따른 선택적 체크포인팅을 잘 활용하면, 5% 미만의 FLOPs 오버헤드로 메모리를 5배 절약할 수 있습니다. 그리고 FP8 행렬 곱셈(matmuls), FSDP 오프로딩(offload), 전문가 병렬(expert-parallel) MoE와 결합할 때 이는 매우 중요합니다. 메모리나 낭비되는 계산 모두를 감당할 수 없기 때문입니다.

## 개념

### 역전파가 실제로 필요한 것

`output = layer(input)`. 역전파는 `grad_input`과 `grad_params`를 필요로 합니다. 이를 계산하기 위해 다음이 필요합니다:

- `input` (선형 레이어의 경우 `grad_params = input.T @ grad_output` 계산용)
- 활성화 함수 도함수 중간값 (ReLU/GELU/softmax의 도함수는 활성화 값에 의존)

순전파 과정은 자동 미분 그래프에 이러한 값을 자동으로 저장합니다. 모든 `tensor.retain_grad()`와 입력이 필요한 연산은 참조를 유지합니다.

### 순진한 전체 체크포인팅

네트워크를 `N`개의 세그먼트로 분할합니다. 순전파 중에는 각 세그먼트의 *입력*만 저장합니다. 역전파 시 중간값이 필요하면 해당 세그먼트의 순전파를 다시 실행하여 중간값을 생성한 후 미분합니다.

예시: 32층 트랜스포머를 1층씩 32개 세그먼트로 분할.

- 메모리: 32개의 레이어 입력(작음) vs 32 * (레이어당 활성화 용량)(큼).
- 추가 계산: 세그먼트당 1회 추가 순전파, 즉 총 순전파 FLOPs가 ~33% 증가 (역전파는 순전파의 2배이므로 전체 단계는 1 + 1 + 2 = 4 단위, 원래는 1 + 2 = 3 단위).

이것은 Chen et al. 2016의 원래 레시피입니다: 메모리와 계산 균형을 위해 `sqrt(L)` 레이어마다 체크포인트를 추가합니다. L=64인 경우 8개의 체크포인트가 필요합니다.

### 선택적 체크포인팅 (Korthikanti 2022)

모든 활성화가 동일한 비용을 가지지는 않습니다. 어텐션 소프트맥스 출력은 `B*L*L*heads`이며 시퀀스 길이에 따라 *제곱*으로 증가합니다. FFN 은닉 활성화는 `B*L*4d`이며 선형적으로 증가합니다. 긴 시퀀스의 경우 소프트맥스가 지배적입니다.

선택적 체크포인팅은 저장이 저렴한 활성화(선형 투영, 잔차)는 유지하고 비용이 큰 것(어텐션)만 재계산합니다. 재계산에 드는 FLOPs는 최소화하면서 O(L^2) 메모리를 절약합니다.

Megatron-Core는 이를 "선택적" 활성화 재계산으로 구현합니다. 2024년 이후의 대부분의 프론티어 훈련 실행에 사용됩니다.

### 오프로딩

재계산 대안: 순전파와 역전파 사이에 CPU RAM으로 활성화를 이동. PCIe 대역폭이 필요하며, 유휴 대역폭이 재계산 비용을 초과할 때 유리합니다. 혼합 전략이 일반적입니다: 일부 레이어는 체크포인팅, 다른 레이어는 오프로딩.

FSDP2는 오프로딩을 1급 옵션으로 제공합니다. GPU가 메모리 병목 상태이지만 CPU-GPU 전송에 여유가 있을 때 오프로딩이 효과적입니다.

### 재계산 비용 모델

`L` 레이어 중 `k` 레이어마다 순진한 체크포인팅을 사용할 때의 단계별 FLOPs:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # 세그먼트당 1회 추가 순전파
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

선택적 체크포인팅에서는 전체 레이어가 아닌 어텐션 커널만 재계산합니다:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 메모리 절약 모델

레이어당 활성화 용량: `A`. `L` 레이어의 총 활성화 메모리: `L * A`.

전체 체크포인팅(세그먼트 크기 1): `L * input_volume`만 저장 (표준 트랜스포머의 경우 `L * 1/10 A` 정도). ~`9 * L * A * 1/10` 절약.

`k` 레이어마다 체크포인팅: `L/k * A` + 활성 세그먼트 내 `k-1` 레이어분 저장.

`k = sqrt(L)`에서 메모리와 재계산 비용이 모두 `sqrt(L)`에 비례 — 균일한 비용의 레이어에 대한 최적의 균형.

### 체크포인팅을 사용하지 않는 경우

- 파이프라인 단계의 가장 내부 레이어는 이미 진행 중이므로 완료해야 합니다.
- 첫 번째와 마지막 레이어가 단계 계산을 지배하는 경우 (트랜스포머에서는 드묾).
- FlashAttention을 사용하는 어텐션 커널 — Flash는 이미 소프트맥스를 빠르게 재계산하므로 추가적인 레이어 수준 체크포인팅은 거의 효과가 없습니다.

### 구현 패턴

1. **함수 래퍼:** `torch.utils.checkpoint.checkpoint(fn, input)`로 세그먼트를 래핑. PyTorch는 `input`만 저장하고 역전파 시 나머지를 재계산합니다.

2. **데코레이터 기반:** 레이어를 체크포인팅 가능으로 표시; 트레이너가 구성 시 어떤 세그먼트를 래핑할지 결정.

3. **수동 명시적 재계산:** 역전파 패스를 직접 작성하고 저장된 입력으로 순전파를 복제하는 `recompute_forward`를 호출.

세 가지 모두 동일한 기능적 결과를 제공합니다. 래퍼가 표준 관용구입니다.

### TP / PP / FP8과의 상호작용

- **텐서 병렬:** 재계산 시 체크포인트 입력을 수집하거나 재분산해야 함; 통신 비용 처리 필요.
- **파이프라인 병렬:** 일반적인 패턴은 각 파이프라인 단계의 순전파를 체크포인팅하여 역순 마이크로배치가 활성화 메모리를 재사용할 수 있도록 하는 것.
- **FP8 재계산:** 재계산 중 업데이트된 amax 히스토리는 원래 순전파와 일치해야 FP8 스케일이 드리프트되지 않음. 대부분의 프레임워크는 스케일을 스냅샷합니다.

## 빌드하기

### 단계 1: 세그먼트가 있는 장난감 모델

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 단계 2: 모든 활성화 값이 필요한 순진한 역전파

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 단계 3: k마다 체크포인트 메모리

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 단계 4: 비용 모델

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 단계 5: 메모리 추정기

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 단계 6: 최적의 세그먼트 크기

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 단계 7: 선택적 체크포인트 결정

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 사용 방법

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — PyTorch의 표준 래퍼. 함수를 감싸며, 입력값만 저장하고 역전파 시 재계산합니다.
- **Megatron-Core 활성화 재계산**: `selective`, `full`, `block` 모드를 지원합니다. 2024+ 프론티어 학습에서 표준입니다.
- **FSDP2 오프로드**: `module.to_empty(device="cpu")`와 FSDP2의 `offload_policy`를 사용하여 활성화 값을 재계산 대신 CPU로 분할 저장합니다.
- **DeepSpeed ZeRO-Offload**: 옵티마이저 상태와 활성화 값을 CPU로 오프로드하며, 체크포인팅을 보완합니다.

## Ship It

이 레슨은 `outputs/prompt-activation-recompute-policy.md`를 생성합니다 — 모델 구성(레이어, 히든, 시퀀스, 배치)과 사용 가능한 GPU 메모리를 입력으로 받아 레이어별 재계산 정책(none / selective / full / offload)을 출력하는 프롬프트입니다.

## 연습 문제

1. 정확성 검증. `model_forward` + `model_backward` (전체 활성화) 대 `model_forward_checkpointed` + `model_backward_checkpointed` (세그먼트)를 실행합니다. 매개변수 그래디언트는 머신 정밀도 수준에서 동일해야 합니다.

2. 세그먼트 크기 `k`를 1부터 `L`까지 스위핑합니다. FLOP 오버헤드와 메모리를 플롯합니다. 곡선의 변곡점을 찾습니다.

3. 선택적 체크포인팅 구현: 어텐션 모듈 입력은 저장하지만 중간 활성화는 저장하지 않습니다. 32층 모델에서 시퀀스 길이 8192에 대해 전체 레이어 체크포인팅 대비 FLOP 오버헤드를 측정합니다.

4. 오프로딩 추가. 세그먼트 입력을 시뮬레이션된 "CPU 버퍼"(별도의 리스트)에 저장합니다. "PCIe 대역폭"을 바이트/시간 단위로 측정하고 오프로딩과 재계산 간의 균형점을 찾습니다.

5. 실제 PyTorch 트랜스포머를 `torch.utils.checkpoint` 적용 전후로 벤치마킹합니다. 메모리(`torch.cuda.max_memory_allocated`를 통해)와 스텝 시간을 측정합니다.

## 핵심 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|----------------------|
| 그래디언트 체크포인팅(Gradient checkpointing) | "순전파를 다시 수행하여 메모리 절약" | 세그먼트 입력만 저장; 역전파 중 중간값을 재계산하여 그래디언트 지원 텐서 획득 |
| 활성화 재계산(Activation recomputation) | "체크포인팅과 동일" | 동일한 기법에 대한 HPC(고성능 컴퓨팅) 분야 명칭 |
| 세그먼트 크기(k) | "체크포인트당 레이어 수" | 중간값이 삭제되고 함께 재계산되는 레이어 수 |
| 선택적 체크포인팅(Selective checkpointing) | "Korthikanti의 기법" | 저장 비용이 큰 활성화(예: 어텐션 소프트맥스)만 재계산; 저렴한 것은 유지 |
| 전체 체크포인팅(Full checkpointing) | "순진한 버전" | 모든 세그먼트에서 모든 레이어의 중간값 재계산 |
| 블록 체크포인팅(Block checkpointing) | "거친 세분화" | 전체 트랜스포머 블록을 체크포인팅; 가장 큰 세분화 단위 |
| FLOP 오버헤드 | "계산 세금" | 단계별 추가 FLOPs = (재계산 FLOPs) / (순전파 + 역전파 FLOPs); 순진 버전 33%, 선택적 5% |
| 활성화 오프로딩(Activation offload) | "CPU로 이동" | 순전파→역전파 간 CPU RAM으로 활성화 이동; 재계산 대안 |
| sqrt-L 규칙 | "고전적 최적값" | 균일한 비용 레이어의 경우 최적 체크포인팅 간격은 √L 레이어 |
| 어텐션-소프트맥스 용량 | "O(L²) 문제" | L² × 헤드 수 × 배치 부동소수점 수; 긴 컨텍스트에서 활성화 메모리 지배적 요소 |

## 추가 자료

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 기울기 체크포인팅(gradient checkpointing)을 공식화한 원본 논문
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 선택적 활성화 재계산(selective activation recomputation) 및 공식적 비용 분석
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 역방향 재물질화(reverse-mode rematerialization)를 통한 대체 상수 메모리 접근법
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 대규모 활성화 오프로딩(activation offload)
- [PyTorch `torch.utils.checkpoint` 문서](https://pytorch.org/docs/stable/checkpoint.html) -- 표준 API
- [Megatron-Core 활성화 재계산 문서](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- 선택적(selective), 전체(full), 블록(block) 모드