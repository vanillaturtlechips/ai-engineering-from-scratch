# 프로덕션 양자화 — AWQ, GPTQ, GGUF K-퀀츠, FP8, MXFP4/NVFP4

> 양자화 포맷은 하드웨어, 서빙 엔진, 워크로드에 따라 달라지는 선택 사항입니다. GGUF Q4_K_M 또는 Q5_K_M은 llama.cpp와 Ollama를 통해 CPU 및 엣지 환경을 소유합니다. GPTQ는 동일한 베이스에 여러 LoRA가 필요할 때 vLLM 내부에서 승리합니다. Marlin-AWQ 커널을 사용하는 AWQ는 INT4에서 7B 클래스 모델로 ~741 tok/s의 최고 Pass@1을 제공하며, 2026년 데이터센터 프로덕션의 기본값이 될 것입니다. FP8은 Hopper, Ada, Blackwell에서 중간 지대를 유지하며, 거의 손실 없이 널리 지원됩니다. NVFP4와 MXFP4(Blackwell 마이크로스케일링)는 공격적이며 블록별 검증이 필요합니다. 두 가지 함정이 팀을 괴롭힙니다: 캘리브레이션 데이터셋은 배포 도메인과 일치해야 하며, KV 캐시는 가중치 양자화와 별개입니다. AWQ의 교훈 "내 모델은 이제 4GB야"는 프로덕션 배치 크기에서 10-30GB의 KV 캐시를 잊어버립니다.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 포맷 간 메모리 및 처리량 비교 장난감 코드)  
**선수 지식:** 10단계 · 13단계 (양자화 기초), 17단계 · 04단계 (vLLM 서빙 내부 구조)  
**소요 시간:** ~75분

## 학습 목표

- 2026년 기준 6가지 프로덕션 양자화 형식과 각 형식의 최적 적용 사례를 설명할 수 있다.
- 하드웨어(CPU vs GPU, Hopper vs Blackwell), 엔진(vLLM, TRT-LLM, llama.cpp), 워크로드(일상적 채팅, 추론, 멀티-LoRA)가 주어졌을 때 적절한 양자화 형식을 선택할 수 있다.
- 선택한 형식에 대해 가중치 메모리 절감량과 KV 캐시 미변경 상태를 계산할 수 있다.
- 도메인 트래픽에서 양자화된 모델 성능을 저하시키는 캘리브레이션 데이터셋의 함정을 설명할 수 있다.

## 문제

양자화(Quantization)는 메모리와 HBM 대역폭을 줄이는데, 이는 디코딩에 정확히 필요한 것입니다. FP16 70B 모델의 가중치 크기는 140GB입니다. 가중치를 INT4로 양자화(AWQ 또는 GPTQ)하면 모델 크기는 35GB가 되어 H100 하나에 KV 캐시를 위한 여유 공간과 함께 적합합니다. 이는 128개의 동시 시퀀스와 2k 컨텍스트에서 KV 캐시만으로도 20-30GB가 필요하기 때문에 중요합니다.

하지만 양자화는 무료가 아닙니다. 공격적인 양자화는 특히 추론 중심 작업에서 품질을 저하시킵니다. 다른 형식들은 다른 엔진과 호환되며, 다른 하드웨어는 다른 정밀도를 네이티브로 지원합니다. 2026년의 형식 다양성(Format Zoo)은 현실이며, 다른 사람의 선택을 복사할 수 없습니다 — 자신의 스택에 기반해 선택해야 합니다.

## 개념

### 6가지 포맷

| 포맷 | 비트 | 적합한 사용처 | 엔진 |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, 엣지, 노트북 | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM에서의 멀티-LoRA | vLLM, TGI |
| AWQ | 4 | 데이터센터 GPU 프로덕션 | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 데이터센터 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 멀티유저 | TRT-LLM |
| NVFP4 | 4 | Blackwell 멀티유저 | TRT-LLM |

### GGUF — CPU/엣지 기본 포맷

GGUF는 파일 포맷이며, 그 자체로 양자화 방식은 아님 — K-양자화 변형(Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0)을 하나의 컨테이너에 묶음. Q4_K_M과 Q5_K_M은 프로덕션 기본값 — 4-5비트에서 BF16에 근접한 품질. llama.cpp가 가장 빠른 CPU 추론 엔진이므로 CPU 또는 엣지 서빙에 최적.

vLLM에서의 처리량 페널티: 7B 기준 ~93 tok/s — 이 포맷은 GPU 커널에 최적화되지 않음. 배포 대상이 CPU/엣지일 때만 GGUF 사용. 그 외에는 권장하지 않음.

### GPTQ — vLLM에서의 멀티-LoRA

GPTQ는 보정 패스를 거치는 사후 학습 양자화 알고리즘. Marlin 커널은 GPU에서 빠르게 동작(비교: Marlin 미사용 GPTQ 대비 2.6배 속도 향상). 7B 기준 ~712 tok/s.

유일한 강점: GPTQ-Int4는 vLLM에서 LoRA 어댑터를 지원. 베이스 모델 + 10-50개의 파인튜닝 변형(각각 LoRA로)을 서빙할 경우 GPTQ가 유일한 선택지. NVFP4는 2026년 초 기준으로 아직 LoRA를 지원하지 않음.

### AWQ — 데이터센터 GPU 기본값

활성화 인지 가중치 양자화. 양자화 시 가장 중요한 ~1% 가중치를 보호. Marlin-AWQ 커널: 순진 구현 대비 10.9배 속도 향상. 7B 기준 ~741 tok/s, INT4 포맷 중 최고 Pass@1.

멀티-LoRA(GPTQ)나 Blackwell FP4(NVFP4)가 필요하지 않다면 새로운 GPU 서빙에 AWQ를 선택.

### FP8 — 신뢰할 수 있는 중간 옵션

8비트 부동소수점. 거의 무손실. 널리 지원됨. Hopper Tensor Cores는 FP8을 네이티브로 가속. Blackwell도 상속. 품질이 절대적일 때(추론, 의료, 코드 생성) 2026년 안전한 기본값. 메모리 절약은 INT4의 절반이지만 품질 위험은 훨씬 낮음.

### MXFP4 / NVFP4 — Blackwell 공격적 옵션

마이크로스케일링 FP4. 각 가중치 블록에 고유한 스케일 팩터 적용. Blackwell Tensor Cores에서 하드웨어 가속되는 공격적 양자화. FP8 대비 토큰당 바이트 수 절반 — Phase 17 · 07에서의 경제적 이점.

주의 사항:
- 아직 LoRA 미지원(2026년 초 기준).
- 추론 중심 작업에서 품질 저하 가시적.
- 모델별로 평가 세트에서 검증 필수.

### 보정 함정

AWQ와 GPTQ는 보정 데이터셋(C4 또는 WikiText)이 필요. 도메인 특화 모델(코드, 의료, 법률)의 경우 일반 웹 텍스트로 보정하면 알고리즘이 보호해야 할 가중치를 잘못 판단. HumanEval에서 Pass@1이 몇 포인트 하락할 수 있음.

해결책: 도메인 내 데이터로 보정. 수백 개의 도메인 샘플이면 충분. 배포 전 평가 세트에서 테스트.

### KV 캐시 함정

AWQ는 가중치를 4비트로 축소. KV 캐시는 별도로 FP16/FP8 유지. AWQ를 적용한 70B 모델 예시:

- 가중치: ~35 GB(140 GB에서 INT4로 축소).
- 128 동시 요청 × 2k 컨텍스트의 KV 캐시: ~20 GB.
- 활성화: ~5 GB.
- 총계: ~60 GB — H100 80GB에 적합.

"모델을 4GB로 양자화했다"는 생각은 나머지 30-50GB를 간과. HBM 예산을 종합적으로 계획.

별도로, KV 캐시 양자화(FP8 KV 또는 INT8 KV)는 주의 정확도 직접 영향과 트레이드오프가 있는 별도 선택 — 무료 이득이 아님.

### AWQ INT4는 추론에 위험

체인 오브 사고, 수학, 긴 컨텍스트의 코드 생성 — 공격적 양자화로 인해 가시적 성능 저하. AWQ INT4는 MATH에서 ~3-5점 손실. 추론 중심 작업에는 FP8 또는 BF16 사용; 메모리 비용 감수.

### 2026년 선택 가이드

- CPU/엣지 서빙: GGUF Q4_K_M. 완료.
- GPU 서빙, 일반 채팅, LoRA 없음: AWQ.
- GPU 서빙, 멀티-LoRA: Marlin이 적용된 GPTQ.
- 추론 작업: FP8.
- Blackwell 데이터센터, 검증된 품질: NVFP4 + FP8 KV.
- 애매한 경우: 각 후보 포맷에 대해 1,000개 샘플 평가 실행.

## 사용 방법

`code/main.py`는 다양한 모델 크기에 대해 6가지 형식(weights + KV + 활성화)의 메모리 점유량(footprint)과 상대적 처리량(throughput)을 계산합니다. KV 캐시가 지배적인 영역, 가중치 압축이 효과적인 영역, 그리고 FP8이 안전한 선택인 영역을 보여줍니다.

## Ship It

이 레슨은 `outputs/skill-quantization-picker.md`를 생성합니다. 주어진 하드웨어, 모델 크기, 워크로드 유형, 품질 허용 오차를 기반으로 포맷을 선택하고 캘리브레이션/검증 계획을 생성합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 128 동시 처리 및 2k 컨텍스트 조건에서 70B 모델의 각 포맷별 총 HBM을 계산하세요. 어떤 포맷이 H100 80GB 하나에 적합합니까?
2. 7B 코딩 모델이 있습니다. 포맷을 선택하고 근거를 설명하세요. 품질 허용치에 대한 가정이 틀렸다면, 복구 경로는 무엇입니까?
3. 의료 도메인 모델에 대해 AWQ 보정에 필요한 보정-데이터셋 크기를 계산하세요. 왜 더 많은 데이터가 항상 좋은 것은 아닙니까?
4. Marlin-AWQ 커널 논문 또는 릴리스 노트를 읽고, AWQ가 7B에서 741 tok/s를 달성하는 반면 RAW GPTQ는 ~712 tok/s에 그치는 이유를 3문장으로 설명하세요.
5. AWQ 가중치와 FP8 KV 캐시를 결합하는 것이 BF16 KV 유지보다 나은 경우는 언제입니까?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| GGUF | "llama.cpp 형식" | K-quant 변형체를 번들링하는 파일 형식; CPU/엣지 기본값 |
| Q4_K_M | "Q4 K M" | 4비트 K-quant 중간; 프로덕션 GGUF 기본값 |
| GPTQ | "gee pee tee q" | 훈련 후 INT4 보정; vLLM에서 LoRA 지원 |
| AWQ | "a w q" | 활성화 인식 INT4; Marlin 커널; INT4에서 최고 Pass@1 |
| Marlin 커널 | "빠른 INT4 커널" | Hopper용 INT4 커스텀 CUDA 커널; 10배 속도 향상 |
| FP8 | "8비트 부동소수점" | Hopper/Ada/Blackwell의 안전한 정밀도 기본값 |
| MXFP4 / NVFP4 | "마이크로스케일링 4" | 블록별 스케일 팩터를 가진 Blackwell 4비트 FP |
| 보정 데이터셋 | "cal 데이터" | 양자화 파라미터 선택에 사용되는 입력 텍스트; 도메인과 일치해야 함 |
| KV 캐시 양자화 | "KV INT8" | 가중치와 별도 선택; 어텐션 정확도에 영향 |

## 추가 자료

- [VRLA Tech — LLM 양자화 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — 비교 벤치마크.
- [Jarvis Labs — vLLM 양자화 완전 가이드](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — 포맷별 처리량 수치.
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — 포맷별 선택 가이드.
- [vLLM 문서 — 양자화](https://docs.vllm.ai/en/latest/features/quantization/index.html) — 지원 포맷 및 플래그.
- [AWQ 논문 (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — 원본 AWQ 공식.
- [GPTQ 논문 (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — 원본 GPTQ 공식.