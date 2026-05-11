# 캡스톤 07 — 엔드투엔드 파인튜닝 파이프라인 (데이터에서 SFT, DPO, 서빙까지)

> 자체 데이터로 훈련되고, 자체 선호도에 맞춰 DPO-정렬된 8B 모델, 양자화, 추측 디코딩, 측정 가능한 $/1M 토큰으로 서빙. 2026년 오픈 스택은 Axolotl v0.8, TRL 0.15, 반복 작업을 위한 Unsloth, 양자화를 위한 GPTQ/AWQ/GGUF, 서빙을 위한 EAGLE-3가 적용된 vLLM 0.7입니다. 캡스톤은 전체 파이프라인을 재현 가능하게 실행하는 것 — YAML 입력, 서빙 엔드포인트 출력 — 이며, 2026 모델 개방성 프레임워크 하에 모델 카드를 공개합니다.

**유형:** 캡스톤  
**언어:** Python (파이프라인), YAML (설정), Bash (스크립트)  
**선수 조건:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM 엔지니어링), Phase 17 (인프라), Phase 18 (안전성)  
**수행 단계:** P2 · P3 · P7 · P10 · P11 · P17 · P18  
**소요 시간:** 35시간

## 문제

2026년의 모든 진지한 AI 팀은 파인튜닝 파이프라인을 항상 준비해 둡니다. 최첨단 기본 모델을 출시하기 때문이 아니라, 다운스트림 적응 — 도메인별 SFT, 레이블된 선호도에 대한 DPO, 추측 디코딩을 위한 증류된 초안, EAGLE-3를 활용한 서빙 — 에서 측정 가능한 성과가 나오기 때문입니다. Axolotl v0.8은 멀티-GPU SFT 설정을 처리합니다. TRL 0.15는 DPO와 GRPO를 처리합니다. Unsloth는 빠른 단일-GPU 반복을 가능하게 합니다. EAGLE-3를 탑재한 vLLM 0.7은 품질 저하 없이 디코딩 처리량을 2-3배 증가시킵니다. 툴링은 작동하지만, 핵심은 YAML 파일, 데이터 품질 관리, 평가 규율에 있습니다.

8B 기본 모델(Llama 3.3, Qwen3, Gemma 3)을 작업별 데이터로 SFT 후 DPO를 실행하고, 서빙을 위해 양자화한 뒤 lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro를 기준으로 성능 향상을 측정할 것입니다. 2026 모델 개방성 프레임워크에 따라 모델 카드를 작성할 것입니다. 핵심은 재현성입니다 — 단일 명령어로 전체 파이프라인을 처음부터 끝까지 다시 실행할 수 있어야 합니다.

## 개념

파이프라인은 5단계로 구성됩니다. **데이터**: 중복 제거(MinHash / Datatrove), 품질 필터(Nemotron-CC 스타일 분류기), PII(개인 식별 정보) 제거, 공개 벤치마크 오염 대비 분할-위생 검사. **SFT(지도 미세 조정)**: Axolotl YAML, 8xH100에서 ZeRO-3, 코사인 스케줄, 패킹된 시퀀스, 2-3 에포크. **DPO(직접 선호 최적화) 또는 GRPO(일반화 DPO)**: TRL 설정, 1 에포크, 인간 라벨링 또는 모델 판단 기반 선호 쌍, 베타 튜닝. **양자화**: 배포 유연성을 위한 GPTQ + AWQ + GGUF. **서빙**: EAGLE-3 추측 헤드(vLLM 0.7) 또는 SpecForge(SGLang) 통합, K8s 배포, 대기열 대기 시간 기반 HPA(수평 자동 확장).

검증 결과는 다음과 같이 제공됩니다: SFT 전용 vs SFT+DPO vs SFT+GRPO를 3개의 작업별 벤치마크에서 비교. 서빙 지표: 배치 1/8/32에서의 초당 토큰 수, EAGLE-3 수용률, 100만 토큰당 비용($). 안전성 평가: Llama Guard 4 통과율. 모델 카드: 편향 평가, 재현성 시드, 데이터 라이선스.

## 아키텍처

```
원시 데이터 (HF 데이터셋 + 내부 데이터)
    |
    v
Datatrove 중복 제거 + Nemotron-CC 품질 필터 + PII 제거
    |
    v
데이터 분할 검증 (MMLU-Pro 오염 검사)
    |
    v
Axolotl SFT 설정 (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO 설정       ---> 4xH100, 1 에폭
    |
    v
GPTQ + AWQ + GGUF 양자화
    |
    v
vLLM 0.7 + EAGLE-3 추측 디코딩
    |
    v
K8s 배포, 대기열 대기 기준 HPA
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
모델 카드 (2026 MOF) + 안전성 평가 (Llama Guard 4)
```

## 스택

- **데이터**: 중복 제거를 위한 Datatrove, 품질 분류를 위한 Nemotron-CC classifier, PII 처리를 위한 Presidio
- **기반 모델**: Llama 3.3 8B, Qwen3 14B, 또는 Gemma 3 12B
- **SFT(지도 미세 조정)**: ZeRO-3, Flash Attention 3, 패킹된 시퀀스를 활용한 Axolotl v0.8
- **선호도 튜닝**: DPO 또는 GRPO를 위한 TRL 0.15; 단일 GPU 반복 작업을 위한 Unsloth
- **양자화**: GPTQ (Marlin), AWQ, llama.cpp를 통한 GGUF
- **서빙**: EAGLE-3 추론 디코딩을 지원하는 vLLM 0.7 (또는 SGLang 0.4 + SpecForge)
- **평가**: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- **안전성 평가**: Llama Guard 4, ShieldGemma-2
- **인프라**: Kubernetes + NVIDIA 장치 플러그인, 대기열 대기 메트릭 기반 HPA
- **관측 가능성**: 훈련 시 W&B, 추론 시 Langfuse

## 빌드하기

1. **데이터 파이프라인.** Datatrove dedup를 원시 코퍼스에 실행합니다. Nemotron-CC 스타일 품질 분류기를 적용합니다. Presidio로 PII를 제거합니다. 명시적 시드로 train/val 분할을 작성합니다.

2. **오염 검사.** 모든 검증 분할에 대해 MMLU-Pro, MT-Bench-v2, RewardBench-2 테스트 세트와 MinHash를 계산합니다. 중복이 발견되면 거부합니다.

3. **Axolotl SFT.** ZeRO-3, FA3, 시퀀스 패킹이 포함된 YAML을 사용합니다. 8xH100에서 2-3 에폭 동안 학습합니다. W&B에 로깅합니다.

4. **TRL DPO / GRPO.** SFT 체크포인트를 가져와 선호도 쌍에 대해 DPO를 1 에폭 실행합니다(또는 수학/코드에 검증 가능한 보상을 사용한 GRPO). 베타를 스윕합니다.

5. **양자화.** 세 가지 양자화 모델을 생성합니다: GPTQ-INT4-Marlin, AWQ-INT4, llama.cpp용 GGUF-Q4_K_M. 크기와 명목 처리량을 기록합니다.

6. **추측 디코딩으로 서빙.** Red Hat Speculators를 통해 훈련된 EAGLE-3 초안 헤드가 있는 vLLM 0.7 설정을 사용합니다. 배치 1/8/32에서 수용률과 꼬리 지연 시간을 측정합니다. 동일한 평가 기준에서의 $/1M 토큰을 Anthropic/OpenAI와 비교하여 보고합니다.

7. **평가 매트릭스.** lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro를 base, SFT-only, SFT+DPO, SFT+GRPO에 실행합니다. 표를 생성합니다.

8. **안전성 평가.** 개발 세트에 대한 Llama Guard 4 통과율. ShieldGemma-2 출력 필터.

9. **모델 카드.** MOF 2026 템플릿: 데이터, 학습, 평가, 안전성, 라이선스, 재현성 섹션에 YAML 및 커밋 SHA 포함.

## 사용 방법

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k 중복 제거, 12k 필터링, 280k 수용 (seed=7)
[SFT]     3 에폭, 8xH100, 6시간 12분, 검증 손실 1.42 -> 1.03
[DPO]     1 에폭, beta=0.08, 4xH100, 1시간 40분
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 수용률 0.74, p99 126ms @ 배치 크기=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    모델 카드(model-card.md)가 2026 MOF 하에 생성됨
```

## Ship It

`outputs/skill-finetuning-pipeline.md`는 산출물을 설명합니다. 단일 명령어로 데이터를 SFT(Supervised Fine-Tuning)를 거쳐 DPO( Direct Preference Optimization), 양자화(quantization), 서빙(serving), 평가(evaluation)까지 실행하고, 모델 카드(model card)와 서빙 엔드포인트(served endpoint)를 출력합니다.

| 가중치 | 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 베이스 대비 평가 델타 | 목표 작업(MMLU-Pro, MT-Bench-v2, 작업별)에서의 성능 향상 측정 |
| 20 | 파이프라인 재현성 | 하나의 명령어로 동일한 시드(seed)를 사용해 엔드 투 엔드 재실행 |
| 20 | 데이터 위생 | 중복 제거율(dedup rate), PII(개인 식별 정보) 제거 범위, 오염 검사(contamination check) 통과 |
| 20 | 서빙 효율성 | 배치 크기 1/8/32에서의 토큰/초, EAGLE-3 수용률, 100만 토큰당 비용($/1M tokens) |
| 15 | 모델 카드 + 안전성 평가 | 2026 MOF(모델 공개 프레임워크) 완성도 + Llama Guard 4 통과율 |
| **100** | | |

## 연습 문제

1. 동일한 작업별 벤치마크에서 SFT-only, SFT+DPO, SFT+GRPO를 실행해 보세요. 어떤 선호 최적화 방법이 승리하는지, 그리고 얼마나 큰 차이로 승리하는지 보고하세요.

2. Llama 3.3 8B를 Qwen3 14B로 교체하세요. 동일한 품질 수준에서 $/1M 토큰 비용을 측정하세요.

3. 도메인 데이터와 일반 ShareGPT에서 EAGLE-3 수용률을 측정하세요. 델타(차이)와 이것이 지연 시간 예산에 미치는 영향을 보고하세요.

4. 1%의 오염(훈련 데이터에 MMLU-Pro 정답 유출)을 주입한 후 평가를 다시 실행하세요. MMLU-Pro 정확도가 비현실적으로 상승하는 것을 확인하고, 이를 탐지하는 오염 검사 CI 게이트를 구축하세요.

5. 전체 미세 조정 대안으로 LoRA SFT를 추가하세요. 10배 낮은 메모리에서 품질 격차를 측정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| Axolotl | "SFT 트레이너" | SFT, DPO, 지식 증류를 위한 통합 YAML 기반 트레이너 |
| TRL | "선호도 튜너" | LLM에서 DPO, GRPO, PPO를 위한 Hugging Face 라이브러리 |
| GRPO | "그룹 상대 정책 최적화" | 검증 가능한 보상을 갖춘 DeepSeek R1의 RL 레시피 |
| EAGLE-3 | "추측 디코딩 초안" | N 토큰을 미리 예측하는 초안 헤드; vLLM이 대상 모델로 검증 |
| MOF | "모델 개방성 프레임워크" | 데이터, 코드, 라이선스에 대한 모델 출시 등급 2026 표준 |
| 오염 검사 | "분할 위생" | MinHash 기반 훈련 세트로의 테스트 세트 유출 감지 |
| 수용률 | "EAGLE / MTP 메트릭" | 대상 모델이 수용하는 초안 토큰의 비율 |

## 추가 자료

- [Axolotl 문서](https://axolotl-ai-cloud.github.io/axolotl/) — 참조 SFT / DPO 트레이너
- [TRL 문서](https://huggingface.co/docs/trl) — DPO 및 GRPO 참조 구현
- [Unsloth](https://github.com/unslothai/unsloth) — 단일 GPU 반복 참조
- [DeepSeek R1 논문 (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 방법론
- [vLLM + EAGLE-3 문서](https://docs.vllm.ai) — 참조 서빙 스택
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 대체 추측 디코딩 트레이너
- [모델 개방성 프레임워크 2026](https://isocpp.org/) — 공개 릴리스 평가 표준
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 표준 평가 러너