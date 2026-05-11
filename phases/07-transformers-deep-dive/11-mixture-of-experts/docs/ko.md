# 전문가 혼합(Mixture of Experts, MoE)

> 밀집된 70B 트랜스포머는 모든 토큰에 대해 모든 파라미터를 활성화합니다. 671B MoE는 토큰당 37B만 활성화하고 모든 벤치마크에서 이를 능가합니다. 희소성은 10년 간의 가장 중요한 확장 아이디어입니다.

**유형:** 구축(Build)
**언어:** Python
**사전 요구 사항:** 7단계 · 05 (풀 트랜스포머), 7단계 · 07 (GPT)
**소요 시간:** ~45분

## 문제 정의

밀집(dense) 트랜스포머의 추론 시 FLOPs는 파라미터 수와 동일하며(전진 전달 시 2배), 대규모 밀집 모델을 확장하면 모든 토큰이 전체 계산 비용을 지불해야 합니다. 2024년에는 계산 한계에 부딪혔는데, 의미 있는 성능 향상을 위해서는 토큰당 기하급수적으로 더 많은 FLOPs가 필요했습니다.

전문가 혼합(Mixture of Experts, MoE)은 이러한 관계를 끊습니다. 각 피드포워드 네트워크(FFN)를 `E`개의 독립 전문가(expert)와 토큰당 `k`개 전문가를 선택하는 라우터(router)로 대체합니다. 총 파라미터 수 = `E × FFN_size`. 토큰당 활성 파라미터 수 = `k × FFN_size`. 2026년 일반적인 구성: `E=256`, `k=8`. 저장 공간은 `E`에 비례해 확장되며, 계산량은 `k`에 비례합니다.

2026년 연구 최전선은 거의 전적으로 MoE 기반입니다: DeepSeek-V3(671B 총 파라미터 / 37B 활성 파라미터), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. Artificial Analysis의 독립 리더보드에서 상위 10개 오픈소스 모델 모두 MoE입니다.

## 개념

![MoE 레이어: 라우터가 토큰당 E개의 전문가 중 k개 선택](../assets/moe.svg)

### FFN 교체

밀집 트랜스포머 블록:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 블록:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # 토큰당 E 중 k개 선택
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

모든 전문가는 독립적인 FFN(일반적으로 SwiGLU)입니다. 라우터는 단일 선형 레이어입니다. 각 토큰은 자신의 `k`개 전문가를 선택하고, 그 출력의 게이팅된 혼합을 얻습니다.

### 부하 분산 문제

라우터가 90%의 토큰을 전문가 3으로 보내면 다른 전문가들은 자원을 얻지 못합니다. 세 가지 해결 방법이 시도되었습니다:

1. **보조 부하 분산 손실** (Switch Transformer, Mixtral). 전문가 사용량의 분산에 비례하는 페널티를 추가합니다. 효과적이지만 하이퍼파라미터와 두 번째 그래디언트 신호가 추가됩니다.
2. **전문가 용량 + 토큰 드롭** (초기 Switch). 각 전문가는 최대 `C × N/E` 토큰을 처리하며, 초과 토큰은 레이어를 건너뜁니다. 품질이 저하됩니다.
3. **보조 손실 없는 밸런싱** (DeepSeek-V3). 라우터의 상위-k 선택을 조정하는 학습된 전문가별 편향을 추가합니다. 편향은 훈련 손실 외부에서 업데이트됩니다. 주 목표에 페널티가 없습니다. 2024년의 주요 발전.

DeepSeek-V3의 접근 방식: 각 훈련 단계 후, 모든 전문가에 대해 사용량이 목표보다 높거나 낮은지 확인합니다. 편향을 `±γ`만큼 조정합니다. 선택은 `scores + bias`를 사용하며, 게이팅에 사용되는 전문가 확률은 변경되지 않은 원시 `scores`입니다. 라우팅과 표현을 분리합니다.

### 공유 전문가

DeepSeek-V2/V3는 전문가를 *공유*와 *라우팅*으로 분할합니다. 모든 토큰은 모든 공유 전문가를 통과합니다. 라우팅 전문가는 상위-k를 통해 선택됩니다. 공유 전문가는 공통 지식을 포착하고, 라우팅 전문가는 특화됩니다. V3는 1개의 공유 전문가와 256개의 라우팅 전문가 중 상위-8개를 실행합니다.

### 세분화된 전문가

클래식 MoE (GShard, Switch): 각 전문가는 전체 FFN만큼 넓습니다. `E`는 작고(8–64), `k`도 작습니다(1–2).

현대적 세분화 MoE (DeepSeek-V3, Qwen-MoE): 각 전문가는 더 좁습니다(FFN 크기의 1/8). `E`는 크고(256+), `k`도 큽니다(8+). 총 파라미터는 동일하지만 조합 수는 훨씬 빠르게 증가합니다. `C(256, 8) = 400조`개의 가능한 "전문가"가 토큰당 존재합니다. 품질은 상승하고 지연 시간은 유지됩니다.

### 비용 프로파일

토큰당, 레이어당:

| 구성 | 활성 파라미터 / 토큰 | 총 파라미터 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (밀집) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3는 거의 모든 벤치마크에서 Llama 3 70B(밀집)를 능가하면서 **토큰당 더 적은 활성 FLOPs**를 수행합니다. 더 많은 파라미터 = 더 많은 지식. 더 많은 활성 FLOPs = 토큰당 더 많은 계산. MoE는 이 둘을 분리합니다.

### 주의 사항: 메모리

모든 전문가는 어떤 전문가가 활성화되든 GPU에 상주합니다. 671B 모델은 fp16 가중치를 위해 약 1.3TB의 VRAM이 필요합니다. 최첨단 MoE 배포에는 전문가 병렬화가 필요합니다 — 전문가를 GPU에 분산시키고, 네트워크를 통해 토큰을 라우팅합니다. 지연 시간은 행렬 곱이 아닌 all-to-all 통신에 의해 지배됩니다.

## 구축 방법

`code/main.py`를 참조하세요. 순수 표준 라이브러리(stdlib)로 구현된 컴팩트한 MoE 레이어는 다음을 포함합니다:

- `n_experts=8` SwiGLU-ish 전문가(설명을 위한 각각 하나의 선형 레이어)
- top-k=2 라우팅
- 소프트맥스 정규화 게이팅 가중치
- 전문가별 바이어스를 통한 보조 손실 없는 밸런싱

### 1단계: 라우터

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # 선택된 전문가의 ORIGINAL 점수에 대한 소프트맥스
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

바이어스는 선택에 영향을 주지만 게이트 가중치에는 영향을 주지 않습니다. 이는 DeepSeek-V3의 트릭으로, 바이어스가 모델 예측을 변경하지 않으면서 부하 불균형을 수정합니다.

### 2단계: 라우터를 통해 100개 토큰 실행

어떤 전문가가 얼마나 자주 활성화되는지 추적합니다. 바이어스가 없으면 사용이 편향됩니다. 바이어스 업데이트 루프(과사용 전문가에 `-γ`, 미사용 전문가에 `+γ`)를 사용하면 몇 번의 반복 후 사용이 균일 분포로 수렴합니다.

### 3단계: 파라미터 수 비교

MoE 구성의 "밀집 동등" 값을 출력합니다. DeepSeek-V3 형태: 256개 라우팅 + 1개 공유, 8개 활성화, d_model=7168. 총 파라미터 수는 엄청납니다. 활성 파라미터 수는 밀집 Llama 3 70B의 1/7 수준입니다.

## 사용 방법

HuggingFace 로딩:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026년 프로덕션 추론: vLLM은 MoE 라우팅을 네이티브로 지원합니다. SGLang은 가장 빠른 전문가 병렬 경로를 제공합니다. 둘 다 자동으로 top-k 선택과 전문가 병렬 처리를 관리합니다.

**MoE 선택 시기:**
- 토큰당 추론 비용을 낮추면서 최첨단 품질을 원할 때.
- VRAM/전문가 병렬 인프라가 있을 때.
- 작업 부하가 컨텍스트 중심(긴 문서)이 아닌 토큰 중심(채팅, 코드)일 때.

**MoE를 선택하지 말아야 할 시기:**
- 엣지 배포 — 활성 FLOP에 대해 전체 저장 공간을 지불해야 함.
- 지연 시간에 민감한 단일 사용자 서빙 — 전문가 라우팅이 오버헤드를 추가함.
- 소형 모델(<7B) — MoE의 품질 이점은 계산 임계값(~6B 활성 파라미터) 이상에서만 나타남.

## Ship It

`outputs/skill-moe-configurator.md`를 참조하세요. 이 스킬은 새로운 MoE(Mixture of Experts)에 대해 파라미터 예산, 학습 토큰, 배포 대상을 기반으로 E(전문가 수), k(활성화 전문가 수), 공유 전문가 레이아웃을 선택합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행해 보세요. 보조 손실(auxiliary loss) 없이 편향(bias)이 업데이트되면서 50번의 반복 동안 전문가(expert) 사용이 어떻게 균형을 이루는지 관찰해 보세요.
2. **중간.** 학습된 라우터(router)를 해시 기반 라우터(결정론적, 학습 없음)로 교체해 보세요. 품질(quality)과 균형(balance)을 비교해 보세요. 왜 학습된 라우터가 더 좋을까요?
3. **어려움.** GRPO 스타일의 "롤아웃 일치 라우팅(rollout-matched routing)"(DeepSeek-V3.2 기법)을 구현해 보세요: 추론(inference) 중 어떤 전문가(expert)가 활성화되는지 기록하고, 그래디언트 계산 시 동일한 라우팅을 강제 적용하세요. 간단한 정책 그래디언트(policy gradient) 설정에서 이 기법의 효과를 측정해 보세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 전문가(Expert) | "많은 FFN 중 하나" | 독립적인 피드포워드 네트워크; FFN 계산의 희소 슬라이스에 전용된 파라미터. |
| 라우터(Router) | "게이트" | 각 토큰을 각 전문가(Expert)에 대해 점수화하는 작은 선형 계층; 상위-k 선택. |
| 상위-k 라우팅(Top-k routing) | "토큰당 k개의 활성 전문가" | 각 토큰의 FFN 계산은 게이트 가중치에 따라 정확히 k개의 전문가를 통과. |
| 보조 손실(Auxiliary loss) | "부하 균형 페널티" | 편향된 전문가 사용을 페널티로 적용하는 추가 손실 항. |
| 보조 손실 없는(Auxiliary-loss-free) | "DeepSeek-V3의 트릭" | 라우터의 선택에 대한 전문가별 편향만으로 균형 조정; 추가 그래디언트 없음. |
| 공유 전문가(Shared expert) | "항상 활성화" | 모든 토큰이 통과하는 추가 전문가; 공통 지식 포착. |
| 전문가 병렬화(Expert parallelism) | "전문가별 분할" | 서로 다른 전문가를 다른 GPU에 분산; 네트워크 간 토큰 라우팅. |
| 희소성(Sparsity) | "활성 파라미터 < 총 파라미터" | 비율 `k × expert_size / (E × expert_size)`; DeepSeek-V3의 경우 37/671 ≈ 5.5%. |

## 추가 자료

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 아이디어.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch, 클래식 MoE.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + 보조 손실 없는 MoE + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 편향 기반 밸런싱 논문.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 이 레슨의 라우터가 사용하는 세부 분할 + 공유 전문가 분할.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 원본 공유 전문가 논문.