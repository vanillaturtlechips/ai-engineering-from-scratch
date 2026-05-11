# 분리된 프리필/디코드 — NVIDIA 다이나모 및 llm-d

> 프리필은 계산 집약적(compute-bound)이고, 디코드는 메모리 집약적(memory-bound)입니다. 두 작업을 동일한 GPU에서 실행하면 한 리소스가 낭비됩니다. 분리(disaggregation)는 프리필과 디코드를 별도의 풀로 분할하고, KV 캐시를 NIXL(RDMA/InfiniBand 또는 TCP 폴백)을 통해 전송합니다. NVIDIA 다이나모(GTC 2025 발표, 1.0 GA)는 vLLM/SGLang/TRT-LLM 위에 위치하며, 플래너 프로파일러(Planner Profiler) + SLA 플래너(SLA Planner)가 SLO를 충족하기 위해 프리필:디코드 비율을 자동 조정합니다. NVIDIA는 개발자.nvidia.com(2025-06)에서 GB200 NVL72 + 다이나모를 사용한 DeepSeek-R1 MoE의 중간 지연 시간 영역에서 약 6배 성능 향상을 보고했으며, 다이나모 제품 페이지(개발자.nvidia.com, 미공개)에서는 GB300 NVL72 + 다이나모가 Hopper 대비 최대 50배 MoE 처리량을 광고합니다. "30배" 수치는 블랙웰(Blackwell) 전체 스택 + 다이나모 + DeepSeek-R1 보고서를 종합한 커뮤니티 추정치이며, 정확히 30배를 명시한 단일 출처는 확인되지 않았으므로 방향성 있는 주장으로 간주해야 합니다. llm-d(Red Hat + AWS)는 쿠버네티스 네이티브로, 프리필/디코드/라우터를 독립적인 서비스로 구현하며 역할별 HPA를 지원합니다. llm-d 0.5는 계층적 KV 오프로딩, 캐시 인식 LoRA 라우팅, UCCL 네트워킹, 스케일-투-제로를 추가했습니다. 경제성: 여러 고객 사례를 종합한 내부 분석에 따르면, 동일한 SLA에서 기존 공동 배치 서빙에서 다이나모 기반 분리 서빙으로 전환 시 200만 달러 규모 추론 비용에서 30–40% 절감(즉, 연간 60만–80만 달러 절감)이 가능합니다. 구체적인 200만 달러→60만–80만 달러 수치는 단일 사례 연구가 아닌 내부 종합 자료이므로, 참고 문헌 인용이 아닌 대략적인 기준으로 사용해야 합니다. 짧은 프롬프트(<512 토큰, 짧은 출력)는 전송 비용을 정당화하지 못합니다.

**유형:** 학습
**언어:** Python (표준 라이브러리, 간단한 분리형-공동 배치 시뮬레이터)
**선수 지식:** 17단계 · 04 (vLLM 서빙 내부 구조), 17단계 · 08 (추론 메트릭)
**소요 시간:** ~75분

## 학습 목표

- 프리필(prefill)과 디코드(decode)가 서로 다른 최적의 GPU 할당을 가지는 이유를 설명하고, 공동 배치(colocation) 시 발생하는 리소스 낭비를 정량화하라.
- 분리된 아키텍처(disaggregated architecture)를 도식화하라: 프리필 풀(prefill pool), 디코드 풀(decode pool), NIXL을 통한 KV 전송, 라우터(router).
- 분리 아키텍처가 효율적이지 않은 경우(짧은 프롬프트, 짧은 출력)의 조건을 명시하라.
- NVIDIA 다이나모(Dynamo, 스택-어보브)와 llm-d(쿠버네티스-네이티브)의 차이점을 구분하고, 각각을 운영 환경에 매칭하라.

## 문제

Llama 3.3 70B를 8개의 H100에서 실행할 때, 혼합 워크로드(긴 프롬프트 + 짧은 출력) 환경에서는 대부분의 연산이 프리필(prefill)에 집중되기 때문에 디코딩(decode) 단계에서 GPU가 유휴 상태가 됩니다. 반대로 다른 워크로드(짧은 프롬프트 + 긴 출력) 환경에서는 그 반대의 현상이 발생합니다. 프리필과 디코딩을 동일 GPU에 배치하면 두 경우 모두에 대해 과도한 프로비저닝(over-provisioning)이 발생합니다.

예산 영향: GPU 시간의 20-40%가 잘못된 리소스에 할당되어 낭비됩니다. 메모리 바운드(memory-bound) 디코딩을 실행하기 위해 H100 컴퓨팅 성능을 구매하거나, 컴퓨팅 바운드(compute-bound) 프리필을 실행하기 위해 H100 HBM 대역폭을 구매하는 상황이 발생합니다. 두 경우 모두 비용이 큰 낭비로 이어집니다.

분리(disaggregation) 방식은 프리필과 디코딩을 각각의 병목 현상에 맞춰 크기가 조정된 별도의 풀로 분리합니다. KV 캐시는 고대역폭 인터커넥트(interconnect)를 통해 프리필 풀에서 디코딩 풀로 전송됩니다.

## 개념

### 병목 현상이 다른 이유

**프리필(Prefill)** — 전체 입력 프롬프트에 대해 트랜스포머를 한 번의 순전파로 실행합니다. 행렬 곱셈이 지배적; 계산 중심. H100 FP8은 ~2000 TFLOPS의 유용한 처리량을 제공합니다. 배치 효율성이 좋음 — 한 번의 순전파로 많은 토큰을 처리합니다.

**디코딩(Decode)** — 한 번에 하나의 토큰을 생성하며, 매 반복마다 전체 가중치를 읽습니다. 메모리 대역폭 중심. HBM3은 ~3 TB/s를 제공합니다. 배치 효율성은 높은 동시성에서만 좋음 — 가중치는 배치에 걸쳐 분산됩니다.

**병치(Colocating)** — 두 작업에 최적화된 GPU를 구매합니다. H100은 두 작업 모두에 적합하지만 비용은 동일합니다. 대규모에서는 H100/계산 중심 프리필 풀과 H200/메모리 중심 디코딩 풀 또는 공격적인 양자화를 사용하는 것이 좋습니다.

### 아키텍처

```
            ┌──────────────┐
  요청 → │    라우터    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (프롬프트만)                  │
            ┌──────────────┐    KV 캐시    ┌───────▼──────┐
            │ 프리필 풀   │ ─── NIXL ────► │ 디코딩 풀   │
            │  (계산)     │                │  (메모리)   │
            └──────────────┘                └──────┬───────┘
                                                   │ 토큰
                                                   ▼
                                                 클라이언트
```

NIXL은 NVIDIA의 노드 간 전송 기술입니다. 사용 가능한 경우 RDMA/InfiniBand를 사용하고, 그렇지 않으면 TCP로 대체합니다. 전송 지연은 실제로 발생 — 일반적으로 70B FP8에서 4K-토큰 프롬프트의 KV 캐시에 대해 20-80 ms가 소요됩니다. 이 때문에 짧은 프롬프트는 분리(disaggregation)를 정당화하지 못합니다: 전송 비용이 절감 효과를 초과합니다.

### 다이나모 vs llm-d

**NVIDIA 다이나모(Dynamo)** (GTC 2025 발표, 1.0 GA):
- vLLM, SGLang, TRT-LLM 위에 오케스트레이터로 위치합니다.
- 플래너 프로파일러가 워크로드를 측정하고, SLA 플래너가 프리필:디코딩 비율을 자동 구성합니다.
- Rust 코어, Python 확장성.
- 처리량 향상: NVIDIA는 GB200 NVL72 + 다이나모에서 DeepSeek-R1 MoE에 대해 중간 지연 영역에서 6x를 보고(2025-06, developer.nvidia.com); 커뮤니티 보고의 "최대 30x"는 전체 Blackwell + 다이나모 + DeepSeek-R1 스택에 대한 것이며 단일 출처가 없어 방향성 있는 것으로 취급해야 합니다.
- GB300 NVL72 + 다이나모: 다이나모 제품 페이지에 따르면 Hopper 대비 최대 50x MoE 처리량(날짜 미상, developer.nvidia.com).

**llm-d** (Red Hat + AWS, Kubernetes 네이티브):
- 프리필/디코딩/라우터를 독립적인 Kubernetes 서비스로 구현.
- 큐 깊이(프리필) / KV 활용도(디코딩) 신호를 사용한 역할별 HPA.
- `topologyConstraint packDomain: rack`은 고대역폭 KV 전송을 위해 프리필+디코딩 클러스터를 동일 랙에 패킹합니다.
- llm-d 0.5(2026): 계층적 KV 오프로딩, 캐시 인식 LoRA 라우팅, UCCL 네트워킹, 스케일-투-제로.

관리형 스택-위 오케스트레이터를 원하면 다이나모를, Kubernetes 네이티브 프리미티브와 CNCF 생태계에 전념하면 llm-d를 사용하세요.

### 경제성

내부 복합 데이터(단일 사례 연구 아님 — 크기 조정 기준):

- 병치 서빙에 연간 $2M 추론 비용.
- 다이나모를 사용한 분리(disaggregated)로 전환.
- 동일 요청량, 동일 P99 지연 SLA.
- 보고된 절감액: 연간 $600K–$800K(30–40% 감소).
- 새 하드웨어 없음.

이 수치는 단일 인용 가능한 사례 연구가 아닌 여러 고객 공개를 종합한 것입니다. 가장 가까운 공개 데이터 포인트는 Baseten의 다이나모 KV 라우팅으로 TTFT 2x 더 빠름 / 처리량 61% 증가(2025-10, baseten.co), VAST + CoreWeave의 40–60% KV 적중률에서 토큰당 $ 60–130% 더 많음(2025-12, vastdata.com). 절감 효과는 각 풀의 적절한 크기 조정에서 비롯 — 프리필 중심 워크로드(8K+ 접두사가 있는 RAG)는 균형 잡힌 워크로드보다 더 많은 이점을 얻습니다.

### 분리(disaggregation)를 하지 말아야 할 경우

- 프롬프트 < 512 토큰 및 출력 < 200 토큰: 전송 비용이 이득을 초과.
- 소규모 클러스터(< 4 GPU): 풀 다양성이 충분하지 않음.
- 팀이 역할별 확장을 통해 두 GPU 풀을 운영할 수 없음: 다이나모가 도움이 되지만 간단하지 않음.
- RDMA 패브릭 없음: TCP 전송 비용이 더 큼.

### 라우터는 Phase 17 · 11과 통합

분리된 라우터는 KV-캐시 인식(Phase 17 · 11)입니다. 요청은 접두사를 보유한 디코딩 풀에 도착 — 일치하는 것이 없으면 프리필 → 디코딩으로 흐릅니다. 적중률과 분리(disaggregation)는 상호 작용 — 캐시 인식 라우터는 새로운 프리필이 필요한지 여부를 결정합니다.

### 블랙웰의 MoE에서 실제 숫자가 나옵니다

GB300 NVL72 + 다이나모는 Hopper 기준 대비 50x MoE 처리량을 보여줍니다. MoE 전문가 라우팅은 프리필에서 계산 중심이지만 디코딩(전문가 캐시)에서는 메모리 중심이므로 분리는 이중 승리입니다. 2026년 프론티어 모델 서빙은 MoE 중심(DeepSeek-V3, 미래 GPT-5 변형)입니다.

### 기억해야 할 숫자

벤치마크 수치는 변동 — NVIDIA와 추론 스택은 분기마다 업데이트된 결과를 게시합니다. 인용 전 재확인.

- GB200 NVL72 + 다이나모에서 DeepSeek-R1: 중간 지연 영역에서 기준 대비 ~6x 처리량(2025-06, developer.nvidia.com); 전체 Blackwell + 다이나모 스택에 대한 커뮤니티 "최대 30x" 주장은 단일 출처가 없는 방향성 집계.
- GB300 NVL72 + 다이나모: Hopper 대비 최대 50x MoE 처리량(날짜 미상, developer.nvidia.com).
- 절감 기준(내부 복합, 단일 사례 연구 아님): 동일 SLA에서 연간 $2M 지출 중 $600-800K 절감.
- 분리 임계값: 프롬프트 >512 토큰 + 출력 >200 토큰.
- NIXL을 통한 KV 전송: 70B FP8에서 4K-프롬프트 KV에 대해 20-80 ms.

## 사용 방법

`code/main.py`는 동일 위치(colocated) 및 분리(disaggregated) 서빙 방식을 시뮬레이션합니다. 처리량(throughput), 요청당 비용(cost per request), 그리고 프롬프트 길이 교차점(prompt-length crossover)을 보고합니다.

## Ship It

이 레슨은 `outputs/skill-disaggregation-decider.md`를 생성합니다. 주어진 워크로드와 클러스터를 기반으로 기술 분리(disaggregation) 여부를 결정합니다.

## 연습 문제

1. `code/main.py`를 실행합니다. 어떤 프롬프트 길이에서 분리(disaggregation)가 공동 배치(colocation)를 능가합니까?
2. P99 접두사 길이 8K, 출력 300인 RAG 서비스를 위한 프리필 풀(prefill pool)과 디코드 풀(decode pool)을 설계하십시오.
3. Python 런타임 선호도가 없는 순수 쿠버네티스 환경(pure-Kubernetes shop)에서 Dynamo와 llm-d 중 하나를 선택하십시오.
4. KV 전송 비용 계산: 70B FP8에서 4K 프리필 = ~500MB KV. RDMA 100GB/s에서 전송 = 5ms. TCP 10GB/s = 50ms. 어떤 것이 SLA에 중요합니까?
5. MoE 전문가 라우팅은 KV 접근 패턴을 변경합니다. 토큰마다 다른 전문가를 활성화하는 MoE에서 분리(disaggregation)는 어떻게 동작합니까?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Disaggregated serving | "split prefill/decode" | 각 단계(phase)를 위한 별도의 GPU 풀(pool) |
| NIXL | "NVIDIA transport" | Dynamo의 노드 간 KV 전송(RDMA/TCP) |
| NVIDIA Dynamo | "the orchestrator" | vLLM/SGLang/TRT-LLM을 위한 스택 위 코디네이터 |
| llm-d | "Kubernetes native" | Red Hat + AWS K8s 분산 스택 |
| Planner Profiler | "Dynamo auto-config" | 워크로드 측정 및 풀 비율 구성 |
| SLA Planner | "Dynamo policy" | SLO 충족을 위한 prefill:decode 자동 비율 조정 |
| `packDomain: rack` | "llm-d topology" | 빠른 KV를 위해 동일 랙에 prefill+decode 배치 |
| UCCL | "unified collective" | llm-d 0.5 네트워킹 계층(스케일-투-제로 지원) |
| MoE expert routing | "expert per token" | DeepSeek-V3 패턴; 분산 서비스가 도움 |

## 추가 자료

- [NVIDIA — Dynamo 소개](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Kubernetes에서의 분리된 LLM 추론](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM 분리된 서빙 블로그](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 릴리스 노트](https://github.com/llm-d/llm-d/releases)