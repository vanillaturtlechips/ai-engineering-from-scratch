# 자체 호스팅 서빙 선택 — llama.cpp, Ollama, TGI, vLLM, SGLang

> 2026년 자체 호스팅 추론을 주도하는 4가지 엔진. 하드웨어, 규모, 생태계에 따라 선택. **llama.cpp**는 CPU에서 가장 빠름 — 가장 넓은 모델 지원, 양자화 및 스레딩에 대한 완전한 제어. **Ollama**는 개발자용 노트북 1회 명령 설치 — llama.cpp보다 ~15-30% 느림(Go + CGo + HTTP 직렬화), 프로덕션 유사 부하에서 3배 처리량 차이. **TGI는 2025년 12월 11일 유지보수 모드 진입** — 버그 수정만 진행, vLLM 대비 ~10% 낮은 원시 처리량이지만 역사적으로 최고 수준의 관측 가능성과 HF-생태계 통합. 유지보수 상태로 인해 장기 사용 시 위험 — SGLang 또는 vLLM이 새 프로젝트에 더 안전한 기본 선택. **vLLM**은 범용 프로덕션 기본값 — v0.15.1(2026년 2월)에서 PyTorch 2.10, RTX Blackwell SM120, H200 최적화 추가. **SGLang**은 에이전트형 다중 턴/접두사 중심 전문가 — 400,000개 이상의 GPU가 프로덕션 환경(xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS)에서 사용 중. 하드웨어 제약: CPU 전용 → llama.cpp만 가능. AMD/비-NVIDIA → vLLM만 가능(TRT-LLM은 NVIDIA 전용). 2026년 파이프라인 패턴: 개발 = Ollama, 스테이징 = llama.cpp, 프로덕션 = vLLM 또는 SGLang. GGUF/HF 가중치는 전체 과정에서 동일.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 엔진 결정 트리 탐색기)  
**선수 지식:** 엔진 관련 모든 17단계 레슨(04, 06, 07, 09, 18)  
**소요 시간:** ~45분

## 학습 목표

- 하드웨어(CPU / AMD / NVIDIA Hopper / Blackwell), 규모(1명 사용자 / 100명 / 10,000명), 워크로드(일반 채팅 / 에이전트 / 장문 컨텍스트)에 따라 엔진 선택
- 2026년 TGI 유지보수 모드 상태(2025년 12월 11일)와 새 프로젝트가 vLLM 또는 SGLang으로 편향되는 이유 설명
- GGUF 또는 HF 가중치를 일관되게 사용하는 개발/스테이징/프로덕션 파이프라인 설명
- "CPU 전용"이 llama.cpp를 강제하는 이유와 "AMD"가 TRT-LLM을 제외하는 이유 설명

## 문제 정의

팀이 새로운 자체 호스팅 LLM 프로젝트를 시작합니다. 한 엔지니어는 Ollama를, 다른 엔지니어는 vLLM을, 세 번째 엔지니어는 "TGI는 바로 사용할 수 있지 않나요?"라고 말합니다. 세 명 모두 다른 맥락에서는 옳습니다. 하지만 모든 경우에 옳은 것은 없습니다.

2026년에는 선택 트리가 중요합니다: 하드웨어가 첫 번째, 규모가 두 번째, 워크로드가 세 번째입니다. 그리고 2025년의 특정 사건 — TGI가 12월 11일 유지보수 모드로 전환 — 이 새로운 프로젝트의 기본 선택을 바꿉니다.

## 개념

### 다섯 가지 엔진

| 엔진 | 최적 사용 사례 | 참고 사항 |
|--------|----------|-------|
| **llama.cpp** | CPU / 엣지 / 최소 의존성 / 가장 넓은 모델 지원 | CPU에서 가장 빠름, 전체 제어 가능 |
| **Ollama** | 개발자 노트북, 단일 사용자, 원클릭 설치 | llama.cpp보다 15-30% 느림; 프로덕션 처리량 격차 3배 |
| **TGI** | HF 생태계, 규제 산업 | **유지 관리 모드 2025년 12월 11일** |
| **vLLM** | 범용 프로덕션, 100+ 사용자 | 광범위한 프로덕션 기본값; v0.15.1 2026년 2월 |
| **SGLang** | 에이전트형 다중 턴, 접두사 중심 워크로드 | 400,000+ GPU 프로덕션 |

### 하드웨어 우선 결정

**CPU 전용** → llama.cpp. Ollama도 작동하지만 더 느림. CPU에서 다른 엔진은 경쟁력 없음.

**AMD GPU** → vLLM (AMD ROCm 지원). SGLang도 작동. TRT-LLM은 NVIDIA 전용이므로 제외.

**NVIDIA Hopper (H100 / H200)** → vLLM 또는 SGLang 또는 TRT-LLM. 세 가지 모두 최상위 계층.

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM이 처리량 선두주자 (Phase 17 · 07). vLLM과 SGLang이 그 뒤를 이음.

**Apple Silicon (M 시리즈)** → llama.cpp (Metal). Ollama가 이를 래핑.

### 규모 기반 결정

**1 사용자 / 로컬 개발** → Ollama. 한 번의 명령, 첫 토큰 생성은 수초 내.

**10-100 사용자 / 소규모 팀** → vLLM 단일 GPU.

**100-10k 사용자 / 프로덕션** → vLLM 프로덕션 스택 (Phase 17 · 18) 또는 SGLang.

**10k+ 사용자 / 엔터프라이즈** → vLLM 프로덕션 스택 + 분리형 (Phase 17 · 17) + LMCache (Phase 17 · 18).

### 워크로드 기반 결정

**일반 채팅 / Q&A** → vLLM이 광범위한 기본값으로 승리.

**에이전트형 다중 턴 (도구, 계획, 메모리)** → SGLang의 RadixAttention (Phase 17 · 06)이 압도적.

**RAG + 접두사 재사용 집중** → SGLang.

**코드 생성** → vLLM 적합; SGLang이 캐시 측면에서 약간 우수.

**긴 컨텍스트 (128K+)** → vLLM + 청크 기반 프리필; SGLang + 계층형 KV.

### TGI 유지 관리 함정

Hugging Face TGI는 2025년 12월 11일 유지 관리 모드 진입 — 이후 버그 수정만 진행. 역사적으로: 최상위 관측 가능성, 최고 수준의 HF-생태계 통합 (모델 카드, 안전 도구), vLLM 대비 원시 처리량 약간 뒤처짐.

2026년 신규 프로젝트: TGI를 기본값에서 제외. 기존 TGI 배포는 계속 사용 가능하지만 결국 마이그레이션 필요. SGLang과 vLLM이 더 안전한 기본값.

### 파이프라인 패턴

개발 (Ollama) → 스테이징 (llama.cpp) → 프로덕션 (vLLM). 전체 과정에서 동일한 GGUF 또는 HF 가중치 사용. 엔지니어는 노트북에서 빠르게 반복; 스테이징은 프로덕션 양자화 미러링; 프로덕션은 서빙 대상.

### Ollama 주의 사항

Ollama는 개발에 적합. 공유 프로덕션에는 부적합: Go HTTP 직렬화 오버헤드 추가, 동시성 관리가 vLLM보다 단순, OpenTelemetry 지원 부족. Ollama는 단일 사용자, 원클릭 환경에서 사용하고 공유 환경에서는 vLLM으로 전환.

### 자체 호스팅 vs 관리형 서비스는 별도 결정

Phase 17 · 01 (관리형 하이퍼스케일러), · 02 (추론 플랫폼)에서 관리형 서비스 다룸. 이 레슨은 자체 호스팅을 이미 결정했다고 가정. 자체 호스팅 이유: 데이터 거주성, 커스텀 파인튜닝, 대규모 TCO(총 소유 비용), 호스팅되지 않은 도메인 모델.

### 기억해야 할 숫자

- TGI 유지 관리 모드: 2025년 12월 11일.
- vLLM v0.15.1: 2026년 2월; PyTorch 2.10; Blackwell SM120 지원.
- SGLang 프로덕션 규모: 400,000+ GPU.
- Ollama 처리량 격차 vs llama.cpp: 15-30% 느림; 프로덕션 부하 시 3배 차이.

## 사용 방법

`code/main.py`는 결정 트리 탐색기입니다: 하드웨어 + 규모 + 워크로드 정보를 입력받아 엔진을 선택하고 그 이유를 설명합니다.

## Ship It

이 레슨은 `outputs/skill-engine-picker.md`를 생성합니다. 주어진 제약 조건에 따라 엔진을 선택하고 마이그레이션 계획을 작성합니다.

## 연습 문제

1. `code/main.py`를 하드웨어/규모/워크로드에 맞춰 실행해 보세요. 출력 결과가 직관과 일치하나요?
2. 인프라가 12개의 H100과 8개의 MI300X AMD로 구성되었습니다. 어떤 엔진(engine)을 선택해야 할까요? TRT-LLM(TensorRT-LLM)이 고려 대상에서 제외되는 이유는 무엇인가요?
3. 한 팀이 "우리가 익숙한 기술이기 때문"이라는 이유로 2026년에 TGI(Text Generation Inference)를 사용하려고 합니다. 마이그레이션 필요성을 주장해 보세요.
4. Ollama 개발 환경에서 vLLM 프로덕션 환경으로 전환할 때, 양자화(quantization), 구성(configuration), 관측 가능성(observability) 측면에서 어떤 변화가 발생하나요?
5. P99 접두사 길이(prefix length) 8K와 테넌트 간 높은 재사용률을 가진 RAG(Retrieval-Augmented Generation) 제품을 구축하려고 합니다. 엔진(engine)을 선택하고 Phase 17 · 11 + 18과 함께 스택을 구성해 보세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| llama.cpp | "CPU용" | 가장 넓은 모델 지원, CPU에서 가장 빠름 |
| Ollama | "노트북용" | 한 번의 명령으로 설치, 개발용 처리량 |
| TGI | "HF의 서빙" | 2025년 12월부터 유지보수 모드 |
| vLLM | "기본값" | 2026년 광범위한 프로덕션 기준 |
| SGLang | "에이전트용" | 접두사 중심, RadixAttention |
| TRT-LLM | "NVIDIA 전용" | Blackwell 처리량 리더, NVIDIA 전용 |
| GGUF | "llama.cpp 포맷" | 번들 K-양자화 변형 |
| Production-stack | "vLLM K8s" | 17·18단계 참조 배포 |
| 파이프라인 패턴 | "개발→스테이징→프로덕션" | Ollama → llama.cpp → 동일 가중치에서 vLLM |

## 추가 자료

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — 종합 LLM 추론 엔진 비교](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 2026년 최고의 vLLM 대체 솔루션 10선](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI 유지보수 공지](https://github.com/huggingface/text-generation-inference) — 릴리스 노트.
- [vLLM v0.15.1 릴리스 노트](https://github.com/vllm-project/vllm/releases)