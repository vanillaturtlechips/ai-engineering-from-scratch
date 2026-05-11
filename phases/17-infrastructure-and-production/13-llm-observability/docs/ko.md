# LLM 관측 가능성 스택 선택

> 2026년 관측 가능성 시장은 두 가지 범주로 나뉩니다. 개발 플랫폼(LangSmith, Langfuse, Comet Opik)은 모니터링과 평가(evals), 프롬프트 관리, 세션 재생을 번들로 제공합니다. 게이트웨이/계측 도구(Helicone, SigNoz, OpenLLMetry, Phoenix)는 원격 측정(telemetry)에 중점을 둡니다. Langfuse는 MIT 라이선스 코어에 강력한 오픈소스 균형(50K 이벤트/월 무료 클라우드)을 제공합니다. Phoenix는 Elastic License 2.0 하의 OpenTelemetry 네이티브 도구로, 드리프트(drift)/RAG 시각화에 탁월하지만 지속적인 프로덕션 백엔드는 아닙니다. Arize AX는 제로 카피 Iceberg/Parquet 통합을 사용하여 모놀리식 관측 가능성 대비 100배 저렴하다고 주장합니다. LangSmith은 LangChain/LangGraph 분야에서 선두주자이며, $39/사용자/월 가격 정책을 가지고 있으며, 엔터프라이즈 전용으로 자체 호스팅이 가능합니다. Helicone은 프록시 기반 도구로 15-30분 설정 시간, 100K 요청/월 무료 티어를 제공하지만 에이전트 추적(agent traces) 측면에서는 깊이가 부족합니다. 일반적인 프로덕션 패턴: 게이트웨이(Helicone/Portkey) + 평가 플랫폼(Phoenix/TruLens)을 OpenTelemetry로 통합.

**유형:** 학습
**언어:** Python (표준 라이브러리, 장난감 추적-샘플링 시뮬레이터)
**선수 지식:** 17단계 · 08 (추론 메트릭), 14단계 (에이전트 엔지니어링)
**소요 시간:** ~60분

## 학습 목표

- 개발 플랫폼(통합: 평가 + 프롬프트 + 세션)과 게이트웨이/텔레메트리 도구(트레이스 + 메트릭만 제공)를 구분.
- 6대 주요 도구(Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik)의 라이선스, 가격 정책, 최적 사용 사례를 매핑.
- 게이트웨이 도구와 별도의 평가 플랫폼을 결합할 수 있는 OpenTelemetry-글루 패턴 설명.
- 2026년 비용 차별화 요소(Arize AX의 제로-카피 접근법 vs 모놀리식 인제스트) 명명 및 대략 100배 배수 차이 언급.

## 문제

LLM 기능을 출시했습니다. 작동은 합니다. 하지만 프롬프트 실패, 툴 루프, 지연 시간 저하, 비용 급증, 프롬프트-캐시 적중률 등에 대한 가시성이 전혀 없습니다. "LLM 관측 가능성(LLM observability)"을 구글에 검색하면 세 가지 가격대의 여덟 가지 도구가 모두 같은 문제를 해결한다고 주장합니다.

하지만 이 도구들은 같은 문제를 해결하지 않습니다. **LangSmith**은 "이 LangGraph 실행이 왜 실패했나요?"에 답하고, **Phoenix**는 "내 RAG 파이프라인이 드리프트(drift) 중인가요?"에 답하며, **Helicone**은 "어떤 앱이 토큰을 소모하고 있나요?"에 답합니다. **Langfuse**는 "전체 시스템을 자체 호스팅할 수 있나요?"에 답합니다. 각기 다른 도구, 각기 다른 대상입니다.

도구 선택에는 네 가지 기준이 있습니다:  
- **스택(Stack)**: LangChain? 원시 SDK? 멀티 벤더?  
- **라이선스 허용 범위(License tolerance)**: MIT만? Elastic OK? 상용 라이선스 괜찮음?  
- **예산(Budget)**: 무료 티어? 월 $100? 월 $1,000?  
- **자체 호스팅(Self-host)**: 필수? 있으면 좋음? 절대 안 함?

## 개념

### 두 가지 범주

**개발 플랫폼**은 관측 가능성(observability)을 평가(evals), 프롬프트 관리, 데이터셋 버전 관리, 세션 재생과 함께 번들로 제공합니다. 실험을 실행하고 어떤 프롬프트가 효과적이었는지 확인하며, 새로운 프롬프트를 이전 승자와 데이터셋-회귀 분석합니다. LangSmith, Langfuse, Comet Opik 등이 있습니다.

**게이트웨이/텔레메트리 도구**는 추론 호출(프롬프트, 응답, 토큰, 지연 시간, 모델, 비용)을 계측합니다. Helicone, SigNoz, OpenLLMetry, Phoenix 등이 있습니다. 미니멀리스트 접근 방식으로, OpenTelemetry를 통해 별도의 평가 도구와 결합할 수 있습니다.

### Langfuse — OSS 균형

- 코어 Apache/MIT 라이선스; Docker를 통해 자체 호스팅 가능.
- 클라우드 무료 티어: 월 50K 이벤트. 유료: 팀용 월 $29.
- 평가, 프롬프트 관리, 트레이스, 데이터셋. 개발 플랫폼 4가지 기능을 적절히 커버.
- 적합한 경우: LangSmith급 기능이 필요하지만 자체 호스팅 또는 OSS 라이선스를 유지해야 할 때.

### Phoenix (Arize) — 텔레메트리 우선, OpenTelemetry 네이티브

- Elastic License 2.0; 자체 호스팅이 간단.
- RAG 및 드리프트(drift) 시각화에 탁월. 임베딩 공간 산점도를 기본 제공.
- 지속적 프로덕션 백엔드로 설계되지 않음 — 주로 개발 시간 관측 가능성용.
- 적합한 경우: RAG 파이프라인 개발, 드리프트 디버깅, 프로덕션을 위한 별도 게이트웨이와 페어링.

### Arize AX — 대규모 확장 플레이

- 상용. Iceberg/Parquet를 통한 제로-카피 데이터 레이크 통합.
- 대규모에서 모놀리식 관측 가능성(Datadog급) 대비 약 100배 저렴하다고 주장. 계산: S3에 Parquet로 트레이스 저장; Arize가 직접 읽기.
- 적합한 경우: 일일 1,000만 건 이상 트레이스, 기존 데이터 레이크 보유, Datadog 가격 없이 LLM 특화 대시보드 필요.

### LangSmith — LangChain/LangGraph 우선

- 상용, 사용자당 월 $39. 엔터프라이즈에서만 자체 호스팅.
- LangChain 및 LangGraph 스택에 최적. 해당 스택을 사용하지 않으면 매력적이지 않음.
- 적합한 경우: LangChain에 전념하는 팀, 유료 사용 의향 있음.

### Helicone — 프록시 기반 최소 실행 가능

- `OPENAI_API_BASE`를 Helicone 프록시로 교체하여 15~30분 내 설정.
- MIT 라이선스; 월 100K 요청 무료, 유료 월 $20+.
- 장애 조치(failover), 캐싱, 속도 제한 포함 — 게이트웨이 역할도 수행.
- 에이전트/다단계 트레이스 깊이는 부족.
- 적합한 경우: 빠른 시작, 단일 스택 앱, 게이트웨이 + 관측 가능성 통합 필요.

### Opik (Comet) — OSS 개발 플랫폼

- Apache 2.0, 완전 OSS.
- Comet 유산을 가진 Langfuse와 유사한 기능 세트.
- 적합한 경우: 이미 Comet을 사용 중인 ML 팀, 동일 패널에서 LLM 관측 가능성 필요.

### SigNoz — OpenTelemetry 우선 전체 APM

- Apache 2.0. OpenTelemetry를 통해 일반 APM 및 LLM 처리.
- 적합한 경우: 서비스와 LLM 호출 전반에 걸친 통합 관측 가능성.

### 접착제: OpenTelemetry + GenAI 시맨틱 컨벤션

OpenTelemetry는 2025년 말에 GenAI 시맨틱 컨벤션(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`)을 발표했습니다. OTel을 소비하는 도구는 상호 운용 가능합니다. 프로덕션에서 나타나는 패턴:

1. 모든 LLM 호출에서 GenAI 컨벤션으로 OTel을 방출.
2. 일상적인 운영을 위해 게이트웨이(Helicone/Portkey)로 라우팅.
3. 회귀 분석을 위해 평가 플랫폼(Phoenix/Langfuse)에 이중 전송.
4. Arize AX 또는 DuckDB를 통한 장기 분석을 위해 데이터 레이크(Iceberg)에 보관.

### 함정: 잘못된 계층에서 계측

에이전트 프레임워크 내부에서 계측(예: LangSmith 트레이스 추가)하면 해당 프레임워크에 종속됩니다. HTTP/OpenAI-SDK 계층에서 계측(OpenLLMetry 또는 게이트웨이를 통해)하면 이식성이 높아집니다.

### 샘플링 — 모든 것을 보관할 수 없음

일일 100만 건 이상의 요청에서 전체 트레이스 보관은 LLM 호출 비용보다 더 비쌉니다. 규칙 기반 샘플링: 오류 100%, 고비용 100%, 성공 5%. 집계는 항상 보관; 장기 테일을 위한 원시 데이터 보관.

### 기억해야 할 숫자

- Langfuse 무료 클라우드: 월 50K 이벤트.
- LangSmith: 사용자당 월 $39.
- Helicone 무료: 월 100K 요청.
- Arize AX 주장: 대규모에서 모놀리식 대비 약 100배 저렴.
- OpenTelemetry GenAI 컨벤션: 2025년 출시, 2026년 널리 채택.

## 사용 방법

`code/main.py`는 보존 전략(100% 수집, 샘플링, 샘플링 + 오류) 간 1M-트레이스 하루를 시뮬레이션합니다. 각 전략별 저장 비용과 손실되는 데이터를 보고합니다.

## Ship It

이 레슨은 `outputs/skill-observability-stack.md`를 생성합니다. 주어진 스택, 규모, 예산, 라이선스 상태를 고려하여 적절한 도구(들)을 선택합니다.

## 연습 문제

1. LangChain 팀에서 OSS 자체 호스팅 관측 가능성을 원합니다. Langfuse와 Opik 중 하나를 선택하고 근거를 제시하세요.  
   - **Langfuse** 선택 시: 오픈소스이며 자체 호스팅이 가능하며, LLM 호출 추적 및 분석에 특화되어 있습니다. 사용자 정의 가능한 대시보드와 다양한 통합 기능을 제공합니다.  
   - **Opik** 선택 시: 경량화된 설계로 빠른 배포가 가능하며, 실시간 모니터링에 강점이 있습니다. 다만, 커뮤니티와 생태계가 Langfuse보다 작을 수 있습니다.  

2. Datadog에서 500만 개의 트레이스/일 기준 월 $150,000 견적을 받았습니다. Arize AX의 손익분기점을 계산하세요.  
   - Arize AX의 가격 모델(예: $0.10/1,000 트레이스)을 적용하여 총 비용을 계산합니다.  
   - 예시: 5M 트레이스 × $0.10/1K = $500/월. Datadog 대비 99.67% 절감 효과.  

3. 조직의 지침으로 모든 LLM 호출에 적용해야 할 OpenTelemetry GenAI 속성 세트를 설계하세요.  
   - 필수 속성:  
     - `model.name` (모델 이름, 예: `gpt-4-turbo-2024-04-01`)  
     - `model.provider` (공급자, 예: `openai`, `anthropic`)  
     - `prompt.template_id` (프롬프트 템플릿 ID)  
     - `completion.length` (생성된 토큰 수)  
     - `cost.currency` 및 `cost.amount` (통화 및 비용)  
     - `latency.ms` (응답 지연 시간)  
     - `error.type` (오류 유형, 예: `rate_limit`, `invalid_request`)  

4. Phoenix만으로 프로덕션 환경에 충분한지 논하고, 어떤 경우에 부족한지 설명하세요.  
   - **충분한 경우**: 소규모 트래픽, 단순한 LLM 호출, 빠른 프로토타이핑이 필요할 때.  
   - **부족한 경우**:  
     - 고트래픽 환경에서 자동 확장(autoscaling)이 필요할 때.  
     - 복잡한 오케스트레이션(예: 체인/에이전트 패턴)이 필요할 때.  
     - 보안/인증, 로깅, 모니터링 등 엔터프라이즈급 기능이 필요할 때.  

5. Helicone의 프록시 오버헤드는 20ms입니다. P99 TTFT(First Token Time)가 300ms일 때 허용 가능한지 판단하세요. SLA가 100ms라면 어떻게 될까요?  
   - **P99 TTFT 300ms 기준**: 20ms 오버헤드는 약 6.7%로, 허용 가능할 수 있습니다.  
   - **SLA 100ms 기준**: 20ms는 20% 오버헤드로, SLA를 초과할 위험이 있습니다. 대체 솔루션(예: 직접 통합) 또는 SLA 재협상이 필요할 수 있습니다.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| OpenLLMetry | "OTel for LLMs" | LLM을 위한 오픈소스 OpenTelemetry 계측 |
| GenAI 컨벤션 | "OTel 속성" | LLM 호출을 위한 표준 OTel 속성 이름 |
| LangSmith | "LangChain 관측 가능성" | LangChain 생태계와 번들로 제공되는 상용 플랫폼 |
| Langfuse | "OSS LangSmith" | 유사한 기능 세트를 가진 MIT 라이선스 오픈소스 |
| Phoenix | "Arize 개발 도구" | OpenTelemetry 네이티브 개발/평가 플랫폼 |
| Arize AX | "규모 관측 가능성" | 상용 제로-카피 Iceberg/Parquet 관측 가능성 |
| Helicone | "프록시 관측 가능성" | LLM 원격 측정 + 게이트웨이 기능을 수집하는 HTTP 프록시 |
| Opik | "Comet LLM" | Comet의 Apache 2.0 라이선스 오픈소스 개발 플랫폼 |
| 세션 재생 | "트레이스 재실행" | 도구 호출과 함께 전체 에이전트 세션 재생 |
| 평가(Eval) | "오프라인 테스트" | 레이블이 지정된 데이터셋을 통해 후보 모델/프롬프트 실행 |

## 추가 자료

- [SigNoz — 2026년 최고의 LLM 관측 가능성 도구](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 대체 도구 분석](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Langfuse, LangSmith, Helicone, Phoenix 설정 방법](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 시맨틱 규약](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 문서](https://docs.arize.com/phoenix)
- [Helicone 문서](https://docs.helicone.ai/)