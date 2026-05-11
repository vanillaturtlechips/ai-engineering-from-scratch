# 캡스톤 14 — 추측 디코딩 추론 서버

> vLLM 0.7의 EAGLE-3는 실제 트래픽에서 2.5-3배 처리량을 제공합니다. P-EAGLE(AWS 2026)은 병렬 추측을 더욱 발전시켰습니다. SGLang의 SpecForge는 대규모 초안 헤드를 훈련시켰습니다. Red Hat의 Speculators 허브는 일반적인 오픈 모델에 맞춰 정렬된 초안을 공개했습니다. TensorRT-LLM은 NVIDIA에서 추측 디코딩을 1급 기능으로 구현했습니다. 2026년 프로덕션 서빙 스택은 EAGLE 계열 초안, FP8 또는 INT4 양자화, 큐 대기 HPA를 적용한 vLLM 또는 SGLang입니다. 이 캡스톤은 두 오픈 모델을 기준 처리량 대비 2.5배 이상으로 서빙하고 전체 테일 레이턴시 보고서를 작성하는 것입니다.

**유형:** 캡스톤  
**언어:** Python(서빙), C++/CUDA(커널 검사), YAML(설정)  
**선수 과목:** 3단계(딥러닝), 7단계(트랜스포머), 10단계(LLM 기초), 17단계(인프라)  
**적용 단계:** P3 · P7 · P10 · P17  
**소요 시간:** 30시간

## 문제

Speculative decoding은 2026년에 상용화되었습니다. EAGLE-3 초안 생성기(draft head)는 대상 모델의 은닉 상태(hidden states)에서 훈련되며 N 토큰을 미리 예측합니다. 대상 모델은 단일 패스로 검증합니다. 60-80%의 수용률(acceptance rate)은 2-3배의 종단간 처리량(end-to-end throughput)으로 이어집니다. vLLM 0.7은 이를 네이티브로 통합했습니다. SGLang + SpecForge는 훈련 파이프라인을 제공합니다. Red Hat의 Speculators는 Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B에 정렬된 초안(drafts)을 공개했습니다.

핵심은 모델 자체가 아닌 서빙 운영(serving operations)에 있습니다. 수용률은 트래픽 분포(ShareGPT vs 코드 vs 도메인 데이터)에 따라 변동합니다. 거부 시 꼬리 지연 시간(tail latency)은 추측(speculation) 미사용 시보다 악화됩니다. 따라서 정상 상태 토큰/초(steady-state tokens/sec)뿐만 아니라 여러 배치 크기에서 p99를 보고해야 합니다. 100만 토큰당 비용(cost per 1M tokens) 대 Anthropic/OpenAI API 비교는 신뢰성 확보의 핵심 요소입니다.

## 개념

추측 디코딩(speculative decoding)은 두 계층으로 구성됩니다. **초안(draft)** 모델(EAGLE-3 헤드, n-gram 또는 더 작은 대상 정렬 모델)이 매 단계마다 k개의 후보 토큰을 제안합니다. **대상(target)** 모델은 한 번의 패스로 모든 k를 검증하며, 수락된 접두사는 탐욕적 경로를 대체합니다. 수락률은 초안-대상 정렬 및 입력 분포에 따라 달라집니다.

EAGLE-3는 대부분의 트래픽에서 n-gram 초안보다 성능이 우수합니다. P-EAGLE은 더 깊은 초안 트리를 위해 병렬 추측을 실행합니다. 트레이드오프는 거부 시 P99 지연 시간이 더 높다는 점인데, 이는 검증 패스가 더 크기 때문입니다. 서빙 구성은 배치 크기별 지연 시간을 보고하여 이를 표면화해야 합니다.

배포는 쿠버네티스(Kubernetes) 기반입니다. vLLM 0.7은 GPU 또는 텐서 병렬 샤드당 하나의 복제본을 실행합니다. HPA(Horizontal Pod Autoscaler)는 CPU가 아닌 큐 대기 시간에 따라 자동 확장됩니다. FP8(Marlin) 및 INT4(AWQ) 양자화는 GPU 메모리를 H100/H200 범위 내로 유지합니다. 종단 간 보고서는 처리량(throughput), 수락률(acceptance rate), 배치 1/8/32에서의 p50/p99, 그리고 $/1M 토큰입니다.

## 아키텍처

```
request ingress
    |
    v
vLLM 서버 (0.7) 또는 SGLang (0.4)
    |
    +-- 초안: EAGLE-3 헤드 | P-EAGLE 병렬 | n-gram 폴백
    +-- 대상: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     양자화 FP8-Marlin 또는 INT4-AWQ
    |
    v
검증 통과: 대상 모델을 통해 배치 k 초안 토큰 검증
    |
    v (접두사 수락; 거부된 접미사에 대해 재샘플링)
    v
클라이언트로 토큰 스트림 반환
    |
    v
Prometheus 메트릭: 처리량, 수락률, 큐 대기 시간, 지연 시간 p50/p99
    |
    v
큐 대기 메트릭 기반 HPA
```

## 스택

- 서빙: vLLM 0.7 또는 SGLang 0.4  
- 추측 방법: EAGLE-3 초안 헤드, P-EAGLE 병렬 추측, ngram 폴백  
- 초안 훈련: SpecForge (SGLang) 또는 Red Hat Speculators  
- 대상 모델: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B  
- 양자화: FP8 (Marlin), INT4 AWQ  
- 배포: Kubernetes + NVIDIA 장치 플러그인; 큐 대기 메트릭 기반 HPA  
- 평가: ShareGPT, MT-Bench-v2, GSM8K, HumanEval을 통한 도메인 확산 수용도 측정  
- 참조: TensorRT-LLM 추측 디코딩을 통한 벤더 기준선

## 구축 방법

1. **대상 모델 준비.** Llama 3.3 70B를 선택합니다. Marlin을 통해 FP8로 양자화합니다. vLLM 0.7에 1xH100(또는 2x 텐서 병렬)로 배포합니다.

2. **초안 소스 준비.** Red Hat Speculators에서 정렬된 EAGLE-3 초안 헤드를 가져옵니다(또는 SpecForge를 통해 훈련). vLLM의 추측 디코딩 설정에 로드합니다.

3. **기준 수치 측정.** 추측 활성화 전: 배치 1/8/32에서의 tokens/s, p50/p99 지연 시간, GPU 사용률. 공개합니다.

4. **EAGLE-3 활성화.** 설정 전환 후 동일한 벤치마크를 재실행합니다. 속도 향상, 수용률, p99 꼬리 지연 시간 변화를 보고합니다.

5. **P-EAGLE.** 병렬 추측 활성화; 직렬 EAGLE-3 대비 더 깊은 초안 트리 측정. P-EAGLE이 도움이 되는/안 되는 변곡점을 보고합니다.

6. **도메인 트래픽.** ShareGPT, HumanEval, 도메인 특화 트래픽을 동일 서버에 적용. 분포별 수용률 측정. 초안 이탈 발생 지점 식별.

7. **두 번째 대상 모델.** Qwen3-Coder-30B MoE에서 동일 파이프라인 실행. 초안 생성이 더 복잡함(MoE 라우팅 노이즈). 결과 보고.

8. **K8s HPA.** `queue_wait_ms`를 추적하는 HPA와 함께 K8s에 배포. 부하 3배 시 확장 기능 시연.

9. **비용 비교.** 동일 평가 기준 Anthropic Claude Sonnet 4.7 및 OpenAI GPT-5.4 대비 $/1M 토큰 계산. 공개합니다.

## 사용 방법

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 활성
[decode]    배치 크기=8, 단계당 허용 토큰=3.2, 수용률=0.76
[latency]   첫 토큰 42ms, 전체 응답 980ms (620 토큰)
[cost]      지속적 처리량 기준 출력 토큰 100만 개당 $0.34
```

## Ship It

`outputs/skill-inference-server.md`는 결과물을 설명합니다. 측정 가능한 서빙 스택, 추론적 디코딩(speculative decoding), 전체 벤치마크 보고서, K8s 배포가 포함됩니다.

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 베이스라인 대비 속도 향상 | 두 모델에서 품질 일치 시 2.5배+ 처리량(throughput) |
| 20 | 실제 트래픽 수용률 | 분포별 수용률 보고서 |
| 20 | P99 꼬리 지연(tail-latency) 관리 | 배치 1/8/32에서 추론적 디코딩 적용/미적용 시 p99 |
| 20 | 운영(Ops) | K8s 배포, 큐 대기(queue-wait) 기반 HPA, 원활한 롤아웃 |
| 15 | 보고서 및 방법론 | 변경된 사항과 그 이유에 대한 명확한 설명 |
| **100** | | |

## 연습 문제

1. 초안이 목표 버전보다 한 단계 뒤떨어질 때(예: Llama 3.3 -> 3.4 드리프트) 수용률(acceptance-rate) 저하를 측정하고 모니터링 경고를 구축하세요.

2. ngram-폴백(ngram-fallback) 구현: EAGLE-3 수용률이 임계값 아래로 떨어지면 ngram 초안으로 전환하세요. 신뢰성 개선 사항을 보고하세요.

3. 제어된 MoE 실험 실행: 라우팅 노이즈가 주입된 Qwen3-Coder-30B와 주입되지 않은 버전을 비교하세요. 초안 수용률 민감도를 측정하세요.

4. H200(141GB)으로 확장: 레플리카당 모델 크기 여유 공간(headroom)을 보고하고, 양자화되지 않은 Llama 3.3 70B를 서빙할 수 있는지 확인하세요.

5. 동일한 H100 하드웨어에서 TensorRT-LLM 추측 디코딩(speculative decoding)을 벤치마크하세요. vLLM 대비 성능 우위를 보고하세요.

## 주요 용어

| 용어 | 일반적인 표현 | 실제 의미 |
|------|----------------|------------------------|
| Draft model | "Speculator" | 대상이 검증할 N개의 토큰을 제안하는 소형 모델 |
| EAGLE-3 | "2026 초안 아키텍처" | 대상 은닉 상태에 대해 훈련된 초안 헤드; ~75% 수용률 |
| P-EAGLE | "병렬 예측" | 하나의 대상 패스에서 검증되는 초안 분기 트리 |
| Acceptance rate | "Hit rate" | 재샘플링 없이 수용된 초안 토큰 비율 |
| Quantization | "FP8 / INT4" | GPU 메모리에 더 많은 모델을 적재하기 위한 저정밀도 가중치 |
| Queue wait | "HPA 메트릭" | 추론 시작 전 대기 큐에서 요청이 대기하는 시간 |
| Speculators hub | "정렬된 초안" | 일반적인 오픈 모델을 위한 EAGLE 초안의 Red Hat Neural Magic 허브 |

## 추가 자료

- [vLLM EAGLE 및 P-EAGLE 문서](https://docs.vllm.ai) — 참조 서빙 스택
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 병렬 추측 디코딩 논문 + 통합
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 초안-헤드 훈련 파이프라인
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 정렬된 초안 허브
- [TensorRT-LLM 추측 디코딩](https://nvidia.github.io/TensorRT-LLM/) — 벤더 대체 솔루션
- [Fireworks.ai 서빙 아키텍처](https://fireworks.ai/blog) — 상용 참조
- [EAGLE-3 논문 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 방법론 논문
- [vLLM 저장소](https://github.com/vllm-project/vllm) — 코드 및 벤치마크