# 쿠버네티스에서 GPU 오토스케일링 — 카펜터, KAI 스케줄러, 갱 스케줄링

> 세 가지 계층, 하나가 아닙니다. 카펜터는 동적으로 노드를 프로비저닝합니다(1분 이내, 클러스터 오토스케일러보다 40% 빠름). KAI 스케줄러는 갱 스케줄링, 토폴로지 인식, 계층적 큐를 처리합니다 — 8개 중 7개 노드가 하나의 누락된 GPU를 기다리며 대기하고 리소스를 소모하는 부분적 할당 문제를 방지합니다. 애플리케이션 수준 오토스케일러(NVIDIA 다이너모 플래너, llm-d 워크로드 변형 오토스케일러)는 추론 특화 신호 — 큐 깊이, KV 캐시 활용도 — 에 따라 확장합니다. CPU/DCGM 듀티 사이클이 아닙니다. 고전적인 HPA 함정은 `DCGM_FI_DEV_GPU_UTIL`이 듀티 사이클 측정값이라는 점입니다: 100%는 10개 요청 또는 100개 요청일 수 있습니다. vLLM은 KV 캐시 메모리를 사전 할당하므로 메모리가 스케일 다운을 트리거하지 않습니다. 이 레슨에서는 세 계층을 조합하고 실행 중인 GPU 작업을 추론 중간에 종료하는 기본 카펜터 `WhenEmptyOrUnderutilized` 정책을 피하는 방법을 배웁니다.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 장난감 큐 깊이 오토스케일러 시뮬레이터)  
**선수 지식:** 17단계 · 02 (추론 플랫폼 경제학), 17단계 · 04 (vLLM 서빙 내부 구조)  
**소요 시간:** ~75분

## 학습 목표

- 세 가지 오토스케일링 계층(노드 프로비저닝, 갱 스케줄링, 애플리케이션 수준)을 다이어그램으로 표현하고 각 계층에서 사용되는 도구 이름을 말한다.
- `DCGM_FI_DEV_GPU_UTIL`이 vLLM에 부적절한 HPA 신호인 이유를 설명하고, 두 가지 대체 신호(큐 깊이, KV 캐시 활용도)를 제시한다.
- 갱 스케줄링을 설명하고 KAI 스케줄러가 방지하는 부분 할당 실패 모드(8개 GPU 중 7개 유휴 상태)를 기술한다.
- 실행 중인 GPU 작업을 종료하는 Karpenter 통합 정책(`WhenEmptyOrUnderutilized`)을 명시하고 2026년 안전한 대안을 제시한다.

## 문제

귀하의 팀은 Kubernetes에서 LLM 서빙 서비스를 운영합니다. `DCGM_FI_DEV_GPU_UTIL`을 신호로 HPA(Horizontal Pod Autoscaler)를 설정했습니다. 비즈니스 시간 동안 서비스는 100% 활용도에서 정체됩니다. HPA는 확장되지 않습니다 — 이미 최대라고 판단합니다. 수동으로 레플리카를 추가하면 TTFT(Time To First Token)가 감소합니다. 그러나 HPA는 여전히 확장되지 않습니다. 신호가 잘못된 정보를 제공하고 있습니다.

별개로, 노드 확장을 위해 Cluster Autoscaler를 사용합니다. 2시에 1M 토큰 프롬프트가 도착하면 클러스터는 노드 프로비저닝에 3분을 소비하고 요청이 타임아웃됩니다.

또 다른 문제로, 8개의 GPU가 필요한 70B 모델을 2개 노드에 배포합니다. 클러스터에는 7개의 GPU가 사용 가능하고 1개의 GPU가 3개 노드에 분산되어 있습니다. Cluster Autoscaler는 부족한 1개의 GPU를 위해 노드를 프로비저닝합니다. 7개의 노드가 4분 동안 대기하며 비용을 소모하는 동안 Kubernetes는 마지막 GPU를 준비합니다.

세 가지 계층, 세 가지 다른 실패 모드입니다. 2026년의 GPU 인식 자동 확장은 "HPA를 켜는 것"이 아닙니다. 노드 프로비저닝, 갱 스케줄링(gang scheduling), 애플리케이션 신호 기반 자동 확장을 조합하는 것입니다.

## 개념

### 레이어 1 — 노드 프로비저닝 (Karpenter)

Karpenter는 대기 중인 파드를 감시하고 ~45-60초 내에 노드를 프로비저닝합니다(Cluster Autoscaler는 일반적으로 GPU 노드에 90-120초가 소요됩니다). `NodePool` 제약 조건에 따라 인스턴스 유형을 동적으로 선택합니다. 만약 파드가 8개의 H100을 필요로 하고 클러스터에 일치하는 노드가 없다면, Karpenter는 기존 그룹을 확장하는 대신 직접 노드를 프로비저닝합니다.

**통합 함정(consolidation trap)**: Karpenter의 기본 `consolidationPolicy: WhenEmptyOrUnderutilized`는 GPU 풀에 위험합니다. 실행 중인 GPU 노드를 종료하여 더 저렴한 적절한 크기의 인스턴스로 파드를 마이그레이션할 수 있습니다. 추론 워크로드의 경우 실행 중인 요청을 축출하고 새 노드에서 70B 모델을 다시 로드해야 합니다. 이로 인해 몇 분간의 용량 손실과 요청 실패가 발생합니다.

GPU 풀에 대한 안전한 설정:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

이 설정은 Karpenter가 1시간 후에 완전히 비어 있는 노드만 통합하도록 허용하며, 실행 중인 작업은 절대 축출하지 않습니다.

### 레이어 2 — 갱 스케줄링 (KAI 스케줄러)

KAI 스케줄러(프로젝트명 "Karp"에서 변경)는 기본 kube-scheduler가 처리하지 못하는 작업을 처리합니다:

**갱 스케줄링(gang scheduling)** — 모든 또는 아무것도. 8개의 GPU를 필요로 하는 분산 추론 파드는 8개 모두 함께 시작되거나 전혀 시작되지 않습니다. 이 기능이 없으면 8개 중 7개가 시작되고 무기한 대기하며 비용이 낭비되는 부분 할당 함정이 발생합니다.

**토폴로지 인식** — 어떤 GPU가 NVLink를 공유하는지, 같은 랙에 있는지, InfiniBand가 연결되어 있는지 파악합니다. DeepSeek-V3 67B 텐서 병렬 워크로드는 하나의 NVLink 도메인에 유지되어야 하며, KAI 스케줄러는 이를 존중합니다.

**계층적 큐** — 여러 팀이 우선순위와 할당량을 가지고 동일한 GPU 풀을 경쟁합니다. 팀 A의 프로덕션 작업이 팀 B의 학습 작업에 의해 선점되는 것은 우선순위 규칙이 허용할 경우에만 발생합니다.

KAI는 kube-scheduler와 함께 보조 스케줄러로 배포되며, 워크로드에 어노테이션을 추가하여 사용합니다. Ray와 vLLM 프로덕션 스택 모두 통합되어 있습니다.

### 레이어 3 — 애플리케이션 수준 신호

**HPA 함정**: `DCGM_FI_DEV_GPU_UTIL`은 듀티 사이클 메트릭입니다. 각 샘플링 간격에서 GPU가 작업을 수행했는지 여부를 측정합니다. 100% 사용률은 10개의 동시 요청 또는 100개의 요청을 의미할 수 있으며, 어느 쪽이든 GPU는 바쁩니다. 듀티 사이클 기반 스케일링은 맹목적으로 확장하는 것입니다.

더 나쁜 것은 vLLM 및 유사 엔진이 KV 캐시 메모리를 미리 할당한다는 점입니다(`--gpu-memory-utilization`까지). 메모리 사용량은 요청 1개에서도 90% 근처를 유지합니다. 메모리 기반 HPA는 절대 축소되지 않습니다.

**2026년 대체 신호**:

- 큐 깊이(프리필을 기다리는 요청 수)
- KV 캐시 활용도(활성 시퀀스에 할당된 블록 비율)
- 레플리카별 P99 TTFT(SLA 신호)
- 처리량(초당 모든 SLO를 충족하는 요청 수)

NVIDIA Dynamo Planner와 llm-d 워크로드 변형 자동 확장기(WVA)는 이러한 신호를 소비하고 레플리카를 확장합니다. 이들은 LLM 서빙을 위해 HPA를 완전히 대체합니다.

### 어떤 도구를 언제 사용할까

| 확장 결정 | 도구 |
|----------------|------|
| 노드 추가/제거 | Karpenter |
| 멀티 GPU 작업 스케줄링 | KAI 스케줄러 |
| 레플리카 추가/제거 | Dynamo Planner / llm-d WVA (또는 큐 깊이 기반 커스텀 HPA) |
| GPU 유형 선택 | Karpenter NodePool |
| 저우선순위 선점 | KAI 스케줄러 큐 |

### 분리된 프리필/디코드 작업은 모든 것을 복잡하게 만듭니다

분리된 프리필/디코드 작업(Phase 17 · 17)을 실행하는 경우, 서로 다른 확장 트리거를 가진 두 가지 파드 클래스가 있습니다: 프리필 파드는 큐 깊이에 따라 확장되고, 디코드 파드는 KV 캐시 압력에 따라 확장됩니다. llm-d는 이를 별도의 `Services`로 노출하며 역할별 HPA를 제공합니다. 두 작업 앞에 단일 HPA를 배치하지 마십시오.

### 콜드 스타트도 여기서 중요합니다

콜드 스타트 완화(Phase 17 · 10)는 노드 프로비저닝 시간이 사용자에게 노출되는 지점입니다. Karpenter의 45-60초 워밍업, 20GB 모델 로드, 엔진 초기화를 합치면 처음부터 요청이 처리되기까지 2-5분이 소요됩니다. SLO가 중요한 경로에는 웜 풀(`min_workers=1`)을 유지하거나 애플리케이션 계층에서 Modal 스타일의 체크포인팅을 사용하십시오.

### 기억해야 할 숫자

- Karpenter 노드 프로비저닝: ~45-60초 vs Cluster Autoscaler ~90-120초(GPU 노드).
- KAI 스케줄러는 부분 할당 낭비(7-of-8 함정)를 방지합니다.
- HPA 신호로 `DCGM_FI_DEV_GPU_UTIL` 사용: 고장난 신호; 큐 깊이 또는 KV 활용도 사용.
- Karpenter `WhenEmptyOrUnderutilized`: 실행 중인 GPU 작업 종료. 추론에는 `WhenEmpty + consolidateAfter: 1h` 사용.

## 사용 방법

`code/main.py`는 버스트형 GPU 워크로드에서 3계층 오토스케일러(autoscaler)를 시뮬레이션합니다. 순차적 HPA(듀티 사이클), 큐 깊이 HPA, KAI-갱 스케줄링 기반 스케일링을 비교하며, 충족되지 않은 요청, 유휴 GPU 시간(분), 복합 점수를 보고합니다.  

- **순차적 HPA(naive HPA)**: 듀티 사이클(duty cycle) 기반 스케일링 전략  
- **큐 깊이 HPA(queue-depth HPA)**: 대기 큐 길이 기반 스케일링 전략  
- **KAI-갱 스케줄링**: KAI-갱 스케줄링 통합 스케일링 전략  

성능 지표:  
- **Unmet requests**: 처리되지 않은 요청 수  
- **Idle-GPU minutes**: 유휴 상태의 GPU 누적 시간(분)  
- **Composite score**: 종합 성능 점수 (지표 가중치 반영)

## Ship It

이 레슨은 `outputs/skill-gpu-autoscaler-plan.md`를 생성합니다. 클러스터 토폴로지, 워크로드 형태, SLO(서비스 수준 목표)를 기반으로 3계층 오토스케일링 계획을 설계합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 버스트 워크로드에서 순진한 듀티 사이클 HPA가 드롭하는 요청 중 큐 깊이 HPA가 포착하는 요청은 몇 개인가요? 차이는 어디에서 발생하나요?  
2. H100 SXM5에서 Llama 3.3 70B FP8을 서빙하는 클러스터를 위한 Karpenter NodePool을 설계하세요. `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter`, 그리고 비-GPU 워크로드를 이 노드에서 제외하는 테인트를 명시하세요.  
3. 팀에서 "GPU는 사용 가능하지만 파드가 스케줄링되지 않는다"는 이유로 배포가 Pending 상태에 있다고 보고합니다. 진단하세요 — Karpenter, kube-scheduler, KAI Scheduler 중 어떤 문제인가요? 어떤 메트릭이 이를 확인시켜 주나요?  
4. 분리된 프리필 파드를 자동 확장할 신호와 디코드 파드를 위한 다른 신호를 선택하세요. 두 선택을 정당화하세요.  
5. P99 TTFT > 10s에서 평균 60회/일 요청 드롭 이벤트를 발생시키는 24x7 프로덕션 서비스에서 `WhenEmptyOrUnderutilized` 통합 트랩의 비용을 계산하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Karpenter | "노드 프로비저너" | Kubernetes 노드 자동 확장기; 서브-분 단위 프로비저닝 |
| Cluster Autoscaler | "구형 확장기" | Kubernetes 노드 자동 확장기의 이전 버전; 느림, 그룹 기반 |
| KAI Scheduler | "GPU 스케줄러" | 갱 + 토폴로지 + 큐를 위한 보조 스케줄러 |
| Gang scheduling | "전부 또는 전무" | N개의 파드를 원자적으로 스케줄링하거나 모두 연기 |
| Topology awareness | "랙 인식" | NVLink/IB/랙 배치 기반 파드 배치 |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 사용률" | 듀티 사이클 메트릭; LLM 확장 신호가 아님 |
| Queue depth | "대기 중인 요청" | 프리필 바운드 확장을 위한 올바른 HPA 신호 |
| KV cache utilization | "메모리 압력" | 디코드 바운드 확장을 위한 올바른 HPA 신호 |
| Consolidation | "Karpenter 통합" | 더 저렴한 인스턴스 유형으로 노드 종료 |
| `WhenEmpty + 1h` | "안전한 통합" | 실행 중인 GPU 작업을 축출하지 않는 정책 |

## 추가 자료

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 설계 문서 및 구성 예시.
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — 통합 정책 의미론 및 GPU 안전 기본값.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 확장 신호.
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 통합 패턴.
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 관리형 Kubernetes 전용 지침.
- [llm-d GitHub](https://github.com/llm-d/llm-d) — 워크로드 변형 자동 확장기 설계.