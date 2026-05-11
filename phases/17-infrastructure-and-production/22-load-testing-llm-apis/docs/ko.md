# LLM API 부하 테스트 — k6와 Locust의 한계

> 전통적인 부하 테스터는 스트리밍 응답, 가변 출력 길이, 토큰 수준 메트릭, GPU 포화 상태를 고려해 설계되지 않았습니다. 두 가지 함정이 대부분의 팀을 괴롭힙니다. GIL 함정: Locust의 토큰 수준 측정은 Python GIL 하에서 토크나이징을 실행하며, 이는 높은 동시성 환경에서 요청 생성과 경쟁합니다. 이후 토크나이징 백로그는 보고된 토큰 간 지연 시간을 부풀립니다 — 병목 현상은 서버가 아닌 클라이언트에 있습니다. 프롬프트 균일성 함정: 반복 테스트에서 동일한 프롬프트는 토큰 분포의 한 지점만 테스트합니다. 실제 트래픽은 가변 길이와 다양한 접두사 일치를 가집니다. LLMPerf는 `--mean-input-tokens` + `--stddev-input-tokens`로 이를 해결합니다. 2026년 도구 매핑: 토큰 수준 정확도를 위한 LLM 전용 도구(GenAI-Perf, LLMPerf, LLM-Locust, guidellm); **k6 v2026.1.0** + **k6 Operator 1.0 GA (2025년 9월)** — 스트리밍 인식, Kubernetes 네이티브 분산( TestRun/PrivateLoadZone CRDs 경유), CI/CD 게이트에 최적; Vegeta는 Go 상수율 포화 테스트용; Locust 2.43.3은 스트리밍을 위해 LLM-Locust 확장 기능과 함께만 사용. 부하 패턴: 정상 상태, 램프, 스파이크(오토스케일링 테스트), 소크(메모리 누수).

**유형:** 구축  
**언어:** Python (표준 라이브러리, 현실적인 프롬프트 생성기 + 지연 수집기)  
**사전 요구 사항:** 17단계 · 08 (추론 메트릭), 17단계 · 03 (GPU 오토스케일링)  
**소요 시간:** ~75분

## 학습 목표

- LLM API에 대해 일반적인 부하 테스터가 오류를 발생시키는 두 가지 안티패턴(GIL 트랩, 프롬프트-균일성 트랩)을 설명한다.
- 주어진 목적에 맞는 도구를 선택한다: LLMPerf(벤치마크 실행), k6 + 스트리밍 확장(CI 게이트), guidellm(대규모 합성), GenAI-Perf(NVIDIA 레퍼런스).
- 네 가지 부하 패턴(정상, 램프, 스파이크, 소크)을 설계하고 각각이 포착하는 장애 모드를 명명한다.
- 고정 길이 대신 입력 토큰의 평균 + 표준편차를 사용하여 현실적인 프롬프트 분포를 구축한다.

## 문제

500명의 동시 사용자로 LLM 엔드포인트를 k6 테스트했습니다. 테스트는 성공했습니다. 서비스를 출시했습니다. 실제 200명의 사용자가 접속한 프로덕션 환경에서 서비스가 다운되었습니다 — P99 TTFT(첫 토큰까지의 시간)가 급증했고 GPU가 과부하 상태였습니다.

두 가지 문제가 발생했습니다. 첫째, k6는 500개의 동일한 프롬프트를 전송했습니다 — 요청 병합(request-coalescing)과 접두사 캐싱(prefix caching) 덕분에 마치 500개의 동시 디코딩을 처리하는 것처럼 보였지만 실제로는 하나만 처리하고 있었습니다. 둘째, k6는 스트리밍 응답에서 토큰 간 지연 시간(inter-token latency)을 인간의 경험처럼 추적하지 않습니다. k6는 500개의 토큰이 다양한 간격으로 도착하는 것이 아니라 하나의 HTTP 연결만 인식합니다.

LLM에 대한 부하 테스트는 별도의 전문 분야입니다.

## 개념

### GIL 함정 (Locust)

Locust는 Python을 사용하며 GIL 아래에서 클라이언트 측 토큰화를 실행합니다. 높은 동시성 환경에서 토크나이저는 요청 생성 뒤에 대기열에 쌓입니다. 보고된 토큰 간 지연에는 클라이언트 측 토큰화 백로그가 포함됩니다. 서버가 느리다고 생각하지만, 실제로는 테스트 하네스의 문제입니다.

해결책: LLM-Locust 확장은 토큰화를 별도의 프로세스로 이동시키거나, 컴파일된 언어 하네스(k6, tokenizers.rs를 사용하는 LLMPerf)를 사용합니다.

### 프롬프트 균일성 함정

모든 알려진 부하 테스터는 하나의 프롬프트를 구성할 수 있습니다. 10,000회 반복 루프 테스트에서 매번 정확히 동일한 프롬프트가 전송됩니다. 서버는 매번 동일한 접두사를 보게 되어 — 접두사 캐시 적중률이 100%에 가까워지고, 처리량이 매우 좋아 보입니다.

해결책: 프롬프트 분포에서 샘플링합니다. LLMPerf는 `--mean-input-tokens 500 --stddev-input-tokens 150`을 사용하여 다양한 길이와 내용의 프롬프트를 생성합니다.

### 네 가지 부하 패턴

1. **정상 상태(Steady-state)** — 30-60분 동안 일정한 RPS. 기본 성능 회귀를 감지합니다.
2. **램프(Ramp)** — 15분 동안 RPS를 0에서 목표치까지 선형 증가. 용량 한계점, 워밍업 이상 현상을 감지합니다.
3. **스파이크(Spike)** — 2분 동안 갑작스러운 3-10x RPS 증가 후 복귀. 자동 확장 지연, 큐 포화, 콜드 스타트 영향을 감지합니다.
4. **소크(Soak)** — 4-8시간 동안 정상 상태 유지. 메모리 누수, 연결 풀 드리프트, 관측 가능성 오버플로를 감지합니다.

### 2026년 도구 매핑

**LLMPerf** (Anyscale) — Python이지만 Rust 기반 토큰화를 사용. 평균/표준편차 프롬프트 지원. 스트리밍 인식. 성능 테스트에 가장 적합한 기본 도구.

**NVIDIA GenAI-Perf** — NVIDIA의 참조 도구. Triton 클라이언트 사용; 포괄적인 메트릭 커버리지. ITL(Inference Time Latency)에 TTFT(First Token Time)가 제외됨을 유의. LLMPerf는 TTFT를 포함합니다. 동일한 서버에 대해 두 도구는 서로 다른 TPOT(Token Per Second)를 생성합니다.

**LLM-Locust** (TrueFoundry) — GIL 함정을 해결한 Locust 확장. 익숙한 Locust DSL + 스트리밍 메트릭.

**guidellm** — 대규모 합성 벤치마킹.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (2025년 9월)**:
- k6 자체(Go, 컴파일됨, GIL 없음)에 스트리밍 인식 메트릭 추가.
- k6 Operator는 Kubernetes 네이티브 분산 테스트를 위해 TestRun / PrivateLoadZone CRD 사용.
- CI/CD 게이트 및 SLA 테스트에 가장 적합.

**Vegeta** — Go 기반, k6보다 단순. 일정 속도 HTTP 포화 테스트. LLM 인식은 없지만 게이트웨이/속도 제한 테스트에 적합.

**Locust 2.43.3 기본 버전** — LLM에 대한 GIL 함정 존재. LLM-Locust 확장 사용 시에만 해결.

### CI에서의 SLA 게이트

PR에서 k6를 다음 조건으로 실행:

- 기준 RPS에서 각각 30-50회 반복.
- 게이트 조건: P50/P95 TTFT, 5xx 오류 < 5%, TPOT 임계값 미만.
- 위반 시 빌드 실패.

### 현실적인 프롬프트 분포

실제 트래픽 샘플(보유한 경우) 또는 공개된 분포(예: 채팅용 ShareGPT 프롬프트, 코드용 HumanEval)에서 구축. 평균 + 표준편차를 LLMPerf에 입력. 단일 프롬프트 반복은 반드시 피해야 합니다.

### 기억해야 할 숫자

- k6 Operator 1.0 GA: 2025년 9월.
- k6 v2026.1.0: 스트리밍 인식 메트릭.
- 일반적인 LLMPerf 실행: 동시성 X에서 100-1000개 요청.
- 일반적인 CI 게이트: PR당 30-50회 반복.
- 네 가지 패턴: 정상 상태, 램프, 스파이크, 소크.

## 사용 방법

`code/main.py`는 현실적인 프롬프트 분포를 기반으로 부하 테스트를 시뮬레이션하고, 효과적인 TPOT(Throughput per Output Token)를 측정하며, 균일 프롬프트 함정(uniform-prompt trap)을 보여줍니다.

## Ship It

이 레슨은 `outputs/skill-load-test-plan.md`를 생성합니다. 주어진 워크로드와 SLA(서비스 수준 계약)를 바탕으로 도구를 선택하고 4가지 부하 패턴을 설계합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 균일 분포(uniform distribution) vs 현실적 분포(realistic distribution)를 비교 — 차이는 어디에서 발생하나요?
2. CI 게이트를 위한 k6 스크립트를 작성하세요: 100 동시성(concurrent)에서 TTFT(시간-대-첫-토큰) P95 < 800ms, 실행 시간 5분.
3. Soak 테스트에서 메모리가 시간당 50MB 증가하는 것을 확인했습니다. 가능한 세 가지 원인과 이를 구분할 수 있는 계측(instrumentation) 방법을 제시하세요.
4. 10 RPS에서 100 RPS로 스파이크 테스트를 수행합니다. Karpenter + vLLM 프로덕션 스택(Phase 17 · 03 + 18)이 적용된 경우 예상되는 복구 시간(recovery time)은 얼마인가요?
5. GenAI-Perf에서는 TPOT=6ms로 보고되지만, LLMPerf에서는 동일 서버에서 TPOT=11ms로 보고됩니다. 이 차이를 설명하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| LLMPerf | "LLM 하니스" | Anyscale 벤치마크 도구, 스트리밍 인식 |
| GenAI-Perf | "NVIDIA 도구" | NVIDIA 참조 하니스 |
| LLM-Locust | "LLM용 Locust" | GIL 트랩을 해결한 Locust 확장 |
| guidellm | "합성 벤치마크" | 대규모 합성 도구 |
| k6 Operator | "K8s k6" | CRD 기반 분산 k6 |
| GIL trap | "Python 클라이언트 오버헤드" | 토크나이저 백로그로 인해 보고된 지연 시간 증가 |
| Prompt-uniformity trap | "단일 프롬프트 오류" | 동일한 프롬프트 반복 시 캐시 적중, 처리량 과대평가 |
| Steady-state | "일정한 부하" | N분 동안 평평한 RPS(초당 요청 수) |
| Ramp | "선형 증가" | 0에서 목표치까지 지속 시간 동안 증가 |
| Spike | "버스트 테스트" | 갑작스러운 배수 증가 후 복귀 |
| Soak | "장시간 테스트" | 메모리 누수 검출을 위한 시간 테스트 |

## 추가 자료

- [TianPan — LLM 애플리케이션 부하 테스트](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — LLM 부하 테스트 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — LLM 추론 벤치마킹 소개](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)