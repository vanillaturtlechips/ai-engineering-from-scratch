# vLLM 프로덕션 스택과 LMCache KV 오프로딩

> vLLM의 프로덕션 스택은 라우터, 엔진, 관측성이 함께 구성된 참조 Kubernetes 배포입니다. LMCache는 GPU 메모리에서 KV 캐시를 추출하여 쿼리 및 엔진 간 재사용하는(CPU DRAM, 이후 디스크/Ceph) KV 오프로딩 계층입니다. vLLM 0.11.0 KV 오프로딩 커넥터(2026년 1월 출시)는 커넥터 API(v0.9.0+)를 통해 비동기식 및 플러그인 방식으로 구현됩니다. 오프로딩 지연 시간은 사용자에게 노출되지 않습니다. LMCache는 공유 접두사가 없어도 가치가 있습니다. GPU가 KV 슬롯을 모두 사용한 경우, 선점된 요청을 재계산 없이 CPU에서 복원할 수 있습니다. 16x H100(80GB HBM)에서 4개의 a3-highgpu-4g에 걸쳐 발표된 벤치마크: KV 캐시가 HBM을 초과할 때 네이티브 CPU 오프로딩과 LMCache 모두 처리량을 크게 향상시킵니다. 낮은 KV 풋프린트 환경에서는 모든 구성이 작은 오버헤드로 베이스라인과 일치합니다.

**유형:** 학습
**언어:** Python (표준 라이브러리, 장난감 KV-스필 시뮬레이터)
**선수 지식:** 17단계 · 04 (vLLM 서빙 내부 구조), 17단계 · 06 (SGLang/RadixAttention)
**소요 시간:** ~60분

## 학습 목표

- vLLM 프로덕션 스택 레이어(라우터, 엔진, KV 오프로드, 관측 가능성)를 다이어그램으로 표현.
- KV 오프로딩 커넥터 API(v0.9.0+)와 0.11.0 비동기 경로가 오프로드 지연을 숨기는 방식을 설명.
- LMCache CPU-DRAM이 도움이 되는 경우(KV > HBM)와 오버헤드를 추가하는 경우(KV가 HBM에 적합할 정도로 작은 경우)를 정량화.
- 배포 제약 조건을 고려하여 네이티브 vLLM CPU 오프로드와 LMCache 커넥터 중 선택.

## 문제

vLLM 서빙은 동시성(concurrency)이 증가할 때마다 선점(preemption) 이벤트와 함께 GPU의 HBM 사용률이 100%에 도달합니다. 요청이 제거되고(evicted), 재큐잉(requeued)되며, 1분 내에 동일한 2K-토큰 프롬프트를 네 번 다시 프리필(prefill)합니다. GPU 연산 리소스는 중복된 프리필 작업에 소모되며, 실제 처리량(goodput)은 이론적 처리량(raw throughput)보다 훨씬 낮습니다.

GPU를 추가하는 것은 비용이 선형적으로 증가합니다. HBM을 추가로 확장하는 것은 불가능합니다. 하지만 CPU DRAM은 저렴합니다 — 한 소켓에 512GB 이상이 탑재될 수 있으며, HBM보다 지연 시간(latency)이 몇 배 더 길지만 "일시적으로 워밍업(warm)"된 KV 캐시(KV cache)에는 적합합니다.

LMCache는 KV 캐시를 CPU DRAM으로 추출하여 선점된 요청이 빠르게 복구되도록 하고, 여러 엔진 간 반복되는 접두사(prefix)가 각 엔진에서 다시 프리필하지 않고도 캐시를 공유할 수 있게 합니다.

## 개념

### vLLM 프로덕션 스택

`github.com/vllm-project/production-stack`은 참조 Kubernetes 배포입니다:

- **라우터** — 캐시 인식(Phase 17 · 11). KV 이벤트를 소비합니다.
- **엔진** — vLLM 워커. GPU당 또는 TP/PP 그룹당 하나씩.
- **KV 캐시 오프로드** — LMCache 배포 또는 네이티브 커넥터.
- **관측 가능성** — Prometheus 수집, Grafana 대시보드, OTel 트레이스.
- **제어 평면** — 서비스 검색, 구성, 롤링 업데이트.

Helm 차트 + 운영자로 제공됩니다.

### KV 오프로드 커넥터 API (v0.9.0+)

vLLM 0.9.0은 플러그형 KV 캐시 백엔드를 위한 커넥터 API를 도입했습니다. 엔진은 블록을 커넥터로 오프로드하고, 커넥터는 이를 저장(RAM, 디스크, 오브젝트 스토리지, LMCache)합니다. 요청 시 블록이 필요하면 커넥터가 다시 로드합니다.

vLLM 0.11.0(2026년 1월)은 비동기 오프로드 경로를 추가했습니다. 일반적인 경우 엔진이 오프로드로 차단되지 않고 백그라운드에서 오프로드가 발생합니다. 종단 간 지연 시간과 처리량은 워크로드 형태, KV 캐시 히트율, 시스템 압력에 따라 달라집니다. vLLM의 자체 노트에서는 사용자 정의 커널 오프로드가 낮은 히트율에서 처리량을 저하시킬 수 있으며, 비동기 스케줄링이 추측 디코딩과 알려진 상호작용 문제가 있다고 언급합니다.

### 네이티브 CPU 오프로드 vs LMCache

**네이티브 vLLM CPU 오프로드**: 엔진 로컬. KV 블록을 호스트 RAM에 저장합니다. 구현이 빠르고 네트워크 홉이 없습니다. 엔진 간 경계를 넘지 않습니다.

**LMCache 커넥터**: 클러스터 규모. 블록을 공유 LMCache 서버(CPU DRAM + Ceph/S3 티어)에 저장합니다. 블록은 모든 엔진에서 접근 가능합니다. 16x H100 벤치마크가 공개되었습니다.

단일 엔진의 HBM 압력이 있을 때는 네이티브를 선택합니다. 여러 엔진이 접두사를 공유할 때(RAG에서 공통 시스템 프롬프트, 공유 템플릿이 있는 멀티테넌트)는 LMCache를 선택합니다.

### 벤치마크 동작

4개의 a3-highgpu-4g 테스트 환경에서 16x H100(80GB HBM)을 분산한 결과:

- 낮은 KV 풋프린트(짧은 프롬프트, 낮은 동시성): 모든 구성이 베이스라인과 일치, LMCache는 ~3-5% 오버헤드 추가.
- 중간 풋프린트: LMCache가 엔진 간 접두사 재사용에서 도움을 주기 시작.
- KV가 HBM 초과: 네이티브 CPU 오프로드와 LMCache 모두 처리량을 크게 개선. LMCache는 엔진 간 공유로 더 큰 이득.

### LMCache가 결정적인 경우

- 시스템 프롬프트가 테넌트 간 공유되는 멀티테넌트 서빙.
- 문서 청크가 쿼리 간 반복되는 RAG.
- 동일한 베이스 모델의 파인튜닝 변형(LoRA)에서 베이스 모델 KV 재사용으로 중복 작업 감소.
- 선점 작업이 많은 워크로드: CPU에서 복원이 재-프리필보다 저렴.

### 활성화하지 말아야 할 경우

- 작은 HBM 압력 — 이점 없이 오버헤드만 발생.
- 짧은 컨텍스트(<1K 토큰) — 전송 시간이 재-프리필보다 큼.
- 단일 테넌트 단일 프롬프트 워크로드 — 포착할 재사용 없음.

### 분리된 서빙과의 통합

Phase 17 · 17 분리된 서빙 + LMCache는 시너지 효과: 프리필 풀에서 디코드 풀로의 KV 전송이 사용되지 않으면 LMCache에 저장. 이후 쿼리는 LMCache에서 가져옵니다. Phase 17 · 11 캐시 인식 라우터는 로컬 또는 LMCache 공유 캐시와 일치하는 엔진으로 라우팅할 수 있습니다.

### 기억해야 할 숫자

- vLLM 0.9.0: 커넥터 API 출시.
- vLLM 0.11.0(2026년 1월): 비동기 오프로드 경로; 종단 간 지연 영향은 워크로드, KV 히트율, 시스템 압력에 따라 다름(절대적 보장 아님).
- 16x H100 벤치마크: KV 풋프린트가 HBM을 초과할 때 LMCache가 도움.
- 작은 HBM 압력: 이점 없이 3-5% 오버헤드.

## 사용 방법

`code/main.py`는 LMCache 적용 여부에 따른 선점(preemption)이 빈번한 워크로드를 시뮬레이션합니다. 재-프리페치(re-prefills) 회피 횟수, 처리량(throughput) 향상, HBM 이용률 균형점(break-even HBM utilization)을 보고합니다.

## Ship It

이 레슨은 `outputs/skill-vllm-stack-decider.md`를 생성합니다. 워크로드 형태와 vLLM 배포를 고려하여 네이티브(native) vs LMCache vs 둘 다 해당되지 않는 경우를 결정합니다.

## 연습 문제

1. `code/main.py`를 실행합니다. LMCache가 효과를 발휘하기 시작하는 HBM 사용률(utilization)은 얼마인가요?
2. 한 테넌트가 200개의 쿼리/시간당 6K-토큰 시스템 프롬프트를 공유합니다. 테넌트당 예상 LMCache 절감량을 계산하세요.
3. LMCache 서버는 단일 장애 지점(SPOF)입니다. HA(고가용성) 전략(복제본, 네이티브 폴백)을 설계하세요.
4. LMCache는 Ceph의 HDD에 저장합니다. 70B FP8(500MB) 크기의 4K-토큰 KV에 대해, 읽기 시간 vs 재-프리필(prefill) 시간을 비교하세요.
5. vLLM 0.11.0 비동기 경로가 "무료"인지 여부를 논증하세요 — 오버헤드는 어디에 숨어 있나요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Production-stack | "참조 배포(reference deployment)" | vLLM의 Kubernetes Helm 차트 + 오퍼레이터(operator) |
| Connector API | "KV 백엔드 인터페이스(KV backend interface)" | vLLM 0.9.0+ 플러그 가능한 KV 저장소 인터페이스 |
| Native CPU offload | "엔진 로컬 스필(engine-local spill)" | 동일 엔진의 호스트 RAM에 KV 저장 |
| LMCache | "클러스터 KV 캐시(cluster KV cache)" | CPU DRAM + 디스크 기반 엔진 간 KV 캐시 서버 |
| 0.11.0 async | "비차단 오프로드(non-blocking offload)" | 엔진 스트림 뒤에 숨겨진 오프로드 |
| Preemption | "공간 확보를 위한 축출(evict to make room)" | HBM이 가득 찼을 때 KV 캐시 셔플 |
| Prefix reuse | "동일 시스템 프롬프트(same system prompt)" | 여러 쿼리가 시작 부분 공유; 캐시 히트 |
| Ceph tier | "디스크 계층(disk tier)" | 캐시 계층에서 DRAM 아래의 영구 저장소 |

## 추가 자료

- [vLLM 블로그 — KV 오프로딩 커넥터 (2026년 1월)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM 프로덕션 스택 GitHub](https://github.com/vllm-project/production-stack) — Helm 차트 + 오퍼레이터.
- [엔터프라이즈 규모 LLM 추론을 위한 LMCache (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — 커넥터 구현.
- [vLLM 0.11.0 릴리스 노트](https://github.com/vllm-project/vllm/releases) — 비동기 경로 세부 정보.