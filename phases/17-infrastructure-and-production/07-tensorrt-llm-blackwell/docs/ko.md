# Blackwell에서 FP8 및 NVFP4를 활용한 TensorRT-LLM

> TensorRT-LLM은 NVIDIA 전용이지만 Blackwell에서 뛰어난 성능을 보입니다. Dynamo 오케스트레이션이 적용된 GB200 NVL72에서 SemiAnalysis InferenceX는 2026년 1분기에서 2분기까지 120B 모델에서 토큰 100만 개당 $0.012의 비용을 측정했으며, H100 + vLLM 기준 $0.09/M 대비 7배의 경제적 격차를 보였습니다. 이 스택은 세 가지 부동소수점 체계가 결합된 것입니다: FP8은 KV 캐시와 어텐션 커널에 필수적인 동적 범위를 제공하므로 여전히 중요합니다; NVFP4(4비트 마이크로스케일링)는 가중치와 활성화 값을 처리합니다; 멀티토큰 예측(MTP)과 분리된 프리필/디코드는 추가로 2-3배의 성능 향상을 제공합니다. Day-0 모델 지원은 사후 훈련 변환 없이 FP4 가중치를 직접 로드합니다. 2026년 엔지니어링 팀의 주의점: TRT-LLM은 폐쇄적인 NVIDIA 스택이므로 채택 시 이식성을 희생하고 처리량을 얻게 됩니다. 모델 및 하드웨어 조합에 대한 계산을 수행한 후 결정하세요.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 장난감 FP8/NVFP4 메모리 및 비용 계산기)  
**선수 지식:** 17단계 · 04 (vLLM 서빙 내부 구조), 10단계 · 13 (양자화)  
**소요 시간:** ~75분

## 학습 목표

- 가중치(weights)가 NVFP4에 있을 때에도 KV 캐시(cache)와 어텐션(attention)에 FP8이 여전히 중요한 이유를 설명.
- BF16, FP8, NVFP4 환경에서 프론티어 모델(frontier model)의 HBM 풋프린트(HBM footprint)를 계산하고, 절감 효과가 발생하는 원인을 분석.
- TRT-LLM이 활용하는 Blackwell 전용 기능(FP4 day-0 지원, MTP, 분리된 서빙(disaggregated serving), all-to-all 프리미티브)을 열거.
- Hopper의 vLLM 대비 TRT-LLM의 NVIDIA-lock이 7배 비용 격차를 감수할 만한 가치가 있는 경우를 판단.

## 문제 정의

2026년 추론 경제학의 최전선은 "달러당 토큰 수"입니다. 이 답은 하드웨어 세대(Hopper H100/H200 vs Blackwell B200/GB200), 정밀도(BF16 → FP8 → NVFP4), 서빙 엔진(vLLM vs SGLang vs TRT-LLM), 오케스트레이션(단순 vs 분리형 vs Dynamo)이라는 네 가지 계층적 선택에 따라 달라집니다.

Hopper와 vLLM에서 120B MoE 모델은 백만 토큰당 약 $0.09로 실행됩니다. 반면 Blackwell에서 TRT-LLM + Dynamo를 사용하면 동일한 모델이 백만 토큰당 약 $0.012로 실행되어 7배 더 저렴합니다. 이 차이 중 일부는 하드웨어에서 기인합니다(Blackwell은 Hopper 대비 GPU당 LLM 처리량이 11-15배 높음). 나머지는 스택에서 기인합니다: FP4 가중치, MTP 초안, 분리된 프리필/디코드, MoE 전문가 통신을 위한 NVLink 5 올투올(all-to-all) 등이 있습니다.

NVIDIA 스택 외부에서는 이를 복제할 수 없습니다. 이것이 트레이드오프입니다 — 경제성 대 이식성. 어떤 스택 선택이 차이의 어느 부분을 제공하는지 이해하는 것이 이 강의의 핵심입니다.

## 개념

### KV 캐시에서 FP8이 여전히 기본 단위인 이유

2026년의 흔한 오해: NVFP4가 모든 곳에 적용된다고 가정하는 것. 그렇지 않다. KV 캐시는 넓은 동적 범위를 갖는 어텐션 키와 값을 저장하기 때문에 FP8(8비트 부동소수점)이 필요하다. KV를 FP4로 양자화하면 분포 꼬리가 잘려나가고 어텐션 점수가 붕괴되어 치명적인 정확도 손실이 발생한다. FP8의 지수 비트는 KV 캐시에 필요한 범위를 제공한다.

NVFP4(2025-2026)는 가중치와 활성화 함수에 적용된다. 마이크로스케일링: 각 가중치 블록은 자체 스케일 팩터를 가져 작은 블록들이 텐서별 스케일 손실 없이 서로 다른 동적 범위를 가질 수 있다. 활성화 함수의 경우 FP4가 유지되는데, 이는 활성화 함수가 레이어 내에서 작은 범위를 가지기 때문이다.

일반적인 블랙웰(Blackwell) 구성:

- 가중치: NVFP4(4비트 마이크로스케일링).
- 활성화 함수: NVFP4.
- KV 캐시: FP8.
- 어텐션 누적기: FP32(소프트맥스 안정성).

### TRT-LLM이 사용하는 블랙웰 전용 프리미티브

- **Day-0 FP4 가중치**: 모델 제공업체가 FP4 가중치를 직접 제공; TRT-LLM은 사후 훈련 변환 없이 로드. FP4에 대한 AWQ / GPTQ 단계 불필요.
- **멀티토큰 예측(MTP)**: EAGLE(Phase 17 · 05)과 동일한 개념이지만 TRT-LLM 빌드에 통합.
- **분리형 서빙**: 프리필과 디코드를 별도의 GPU 풀에서 실행, KV 캐시는 NVLink 또는 InfiniBand를 통해 전송. Dynamo(Phase 17 · 20)와 동일한 개념.
- **올투올 통신 프리미티브**: NVLink 5는 Hopper 대비 MoE 전문가 통신 지연 시간을 3배 감소. TRT-LLM의 MoE 커널은 이에 맞춰 최적화.
- **NVFP4 + MXFP8 마이크로스케일링**: 블랙웰 텐서 코어에서 하드웨어 가속 스케일 팩터 처리.

### 외워야 할 수치

- HGX B200에서 TRT-LLM을 통해 GPT-OSS-120B 실행 시 $0.02/백만 토큰.
- GB200 NVL72에서 Dynamo(TRT-LLM 오케스트레이션)를 통해 $0.012/백만 토큰.
- H100 + vLLM은 유사 워크로드에서 약 $0.09/백만 토큰.
- TRT-LLM 업데이트 3개월 만에 2.8배 처리량 증가(2026).
- 블랙웰 대 Hopper 기준 GPU당 LLM 처리량 11-15배 향상.
- MLPerf Inference v6.0(2026년 4월): 블랙웰이 제출된 모든 과제에서 압도적 성능.

### FP4가 실제로 품질에 미치는 영향

NVFP4는 공격적이다. 추론 중심 워크로드(사고 연쇄, 수학, 긴 컨텍스트를 가진 코드 생성)에서 FP4 가중치는 눈에 띄는 성능 저하를 일으킨다. 블록별 보정은 완화하지만 완전히 제거하지는 못한다. 추론 모델을 제공하는 팀은 종종 FP8 가중치 + FP4 활성화 함수를 절충안으로 사용하거나, FP8을 전체적으로 사용하는 H200을 고수한다.

규칙: NVFP4 가중치 적용 전 반드시 평가 세트에서 작업 품질을 검증하라.

### 이것이 NVIDIA 종속 결정인 이유

TRT-LLM은 C++ + CUDA + 폐쇄형 커널로 구성된다. 모델은 특정 GPU SKU에 맞춰 컴파일되어야 한다. AMD, 인텔, ARM은 지원되지 않는다. 인프라 전략이 다중 벤더인 경우, TRT-LLM으로 서빙되는 계층은 제외 — vLLM을 통해 혼합 하드웨어에서 서빙 가능. NVIDIA 전용이라면 7배 성능 차이가 종속 비용을 상쇄한다.

### 2026년 실용적 레시피

연간 1억 달러 이상의 추론 비용이 발생하는 경우, Hopper + vLLM으로 운영하면 7-10배의 성능 향상을 놓친다. 비용 중심 워크로드를 블랙웰 + TRT-LLM + Dynamo로 마이그레이션. 실험 계층은 모델 반복 속도를 위해 H100 + vLLM 유지. NVFP4로 변환된 각 모델의 품질을 검증한 후 프로덕션에 적용.

### 분리형 서빙의 추가 이점

TRT-LLM의 분리형 서빙(프리필과 디코드 풀 분리)은 Phase 17 · 20에서 심층적으로 다뤄진다. 블랙웰에서는 FP4 가중치 × MTP 속도 향상 × 분리형 배치 × 캐시 인식 라우팅이 중첩되어 7배 성능 향상을 달성한다. 이 수치는 전체 스택을 가정한 것이다.

## 사용 방법

`code/main.py`는 H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM 세 가지 스택에 걸쳐 모델의 HBM 풋프린트, 디코딩 처리량(메모리-바운드 영역), $/M-tokens을 계산합니다. 이를 실행하여 각 변경 사항이 기여하는 격차 점유율과 누적 효과를 확인할 수 있습니다.

## Ship It

이 레슨은 `outputs/skill-trtllm-blackwell-advisor.md`를 생성합니다. 워크로드, 모델 크기, 연간 토큰 볼륨을 고려하여 Blackwell + TRT-LLM 스택이 NVIDIA 종속성을 감수할 만한 가치가 있는지 판단합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 120B MoE에서 30% 활성 파라미터를 사용할 때, H100 BF16, H100 FP8, B200 NVFP4/FP8에서 메모리 대역폭 제한 디코드 처리량(throughput)을 계산하세요. 가장 큰 성능 향상은 어디에서 발생하나요?

2. 고객이 H100 + vLLM에 연간 $2M을 지출하고 있습니다. TRT-LLM으로 마이그레이션할 때 12개월 동안 7배의 경제적 격차를 상쇄하려면 몇 개의 Blackwell GPU를 구매해야 하나요?

3. NVFP4 가중치 변환 후 MATH에서 정확도가 3포인트 하락했습니다. 두 가지 복구 경로를 제시하세요: 하나는 품질 우선(FP8 가중치 유지), 다른 하나는 비용 우선(인-도메인 데이터로 캘리브레이션).

4. MLPerf v6.0 추론 결과를 읽으세요. Blackwell과 Hopper 간 성능 격차가 가장 작은 작업은 무엇이며, 그 이유는 무엇인가요?

5. 405B 모델에서 NVFP4 가중치 + FP8 KV 캐시를 128k 컨텍스트로 사용할 때 필요한 HBM 용량을 계산하세요. 단일 GB200 NVL72 노드에 적합합니까?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| FP8 | "8비트 부동소수점" | 8비트 부동소수점; 동적 범위로 인해 KV 캐시 및 어텐션(attention)에 사용 |
| NVFP4 | "4비트 마이크로스케일" | NVIDIA의 4비트 마이크로스케일링(microscaling) FP 포맷; 블랙웰(Blackwell)의 가중치(weights) 및 활성화(activations) |
| MXFP8 | "MX 8비트" | 마이크로스케일링 FP8 변형; 블랙웰 텐서 코어(Tensor Cores)에서 하드웨어 가속 지원 |
| Day-0 FP4 | "FP4 가중치 출시" | 모델 제공업체가 FP4로 이미 변환된 가중치(weights) 공개; 학습 후 변환 단계 불필요 |
| MTP | "멀티토큰 예측" | TRT-LLM의 통합 추측 디코딩(speculative-decoding) 초안(Phase 17 · 05) |
| 분산 추론 | "프리필/디코드 분리" | 프리필(prefill)과 디코드(decode)를 별도의 GPU 풀에서 실행; KV는 NVLink/IB를 통해 전송 |
| All-to-all | "MoE 전문가 통신" | 토큰을 전문가(expert) GPU로 라우팅하는 통신 패턴; NVLink 5는 3배 성능 향상 |
| InferenceX | "SemiAnalysis 추론 벤치마크" | 2026년 업계 표준 토큰당 비용 벤치마크 |

## 추가 자료

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026년 4월 MLPerf 결과.
- [NVIDIA — Blackwell에서의 MoE 추론](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all 및 MoE 커널.
- [TensorRT-LLM 개요](https://nvidia.github.io/TensorRT-LLM/overview.html) — 공식 엔진 문서.
- [NVIDIA — Dynamo 소개](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 상위 분산 오케스트레이션.
- [MLPerf 추론](https://mlcommons.org/benchmarks/inference-datacenter/) — Blackwell 수치를 발표하는 벤치마크 스위트.