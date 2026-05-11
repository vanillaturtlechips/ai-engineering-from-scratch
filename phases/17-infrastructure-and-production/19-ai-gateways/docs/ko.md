# AI 게이트웨이 — LiteLLM, Portkey, Kong AI Gateway, Bifrost

> 게이트웨이는 앱과 모델 공급자 사이에 위치합니다. 핵심 기능은 공급자 라우팅, 폴백, 재시도, 속도 제한, 시크릿 참조, 관측 가능성, 가드레일입니다. 2026년 시장 점유율: **LiteLLM**은 MIT OSS 라이선스로 100개 이상의 공급자를 지원하며 OpenAI와 호환되지만, 약 2000 RPS(8GB 메모리, 공개된 벤치마크에서 캐스케이딩 장애) 수준에서 성능 저하가 발생합니다. Python, <500 RPS, 개발/프로토타이핑에 최적화되어 있습니다. **Portkey**는 제어 평면 중심(가드레일, PII 삭제, 탈옥 감지, 감사 추적)으로 2026년 3월 Apache 2.0 오픈소스로 전환되었으며, 20-40ms 지연 오버헤드가 있고, $49/월 프로덕션 티어가 있습니다. **Kong AI Gateway**는 Kong Gateway 기반으로 구축되었으며, 동일한 12 CPU에서 Kong 자체 벤치마크 결과 Portkey보다 228%, LiteLLM보다 859% 더 빠릅니다. $100/모델/월 가격(Plus 티어 최대 5개)이며, 이미 Kong을 사용 중인 경우 엔터프라이즈에 적합합니다. **Bifrost**(Maxim AI)는 구성 가능한 백오프를 통한 자동 재시도, OpenAI 429 오류 시 Anthropic으로 폴백 기능을 제공합니다. **Cloudflare / Vercel AI 게이트웨이**는 관리형, 제로-옵스, 기본 재시도 기능을 제공합니다. 데이터 거주성은 자체 호스팅 결정을 주도하며, Portkey와 Kong은 OSS + 선택적 관리형으로 중간 위치에 있습니다.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 장난감 게이트웨이-라우팅 시뮬레이터)  
**선수 지식:** 17단계 · 01 (관리형 LLM 플랫폼), 17단계 · 16 (모델 라우팅)  
**소요 시간:** ~60분

## 학습 목표

- 6가지 핵심 게이트웨이 기능(라우팅(routing), 폴백(fallback), 재시도(retries), 속도 제한(rate limits), 시크릿(secrets), 관측 가능성(observability), 가드레일(guardrails))을 열거할 수 있다.
- 2026년 4가지 게이트웨이(LiteLLM, Portkey, Kong AI, Bifrost)를 확장 한계(scale ceilings)와 사용 사례(use cases)에 따라 매핑할 수 있다.
- Kong 벤치마크(Portkey 대비 228%, LiteLLM 대비 859%)를 인용하고, 500 RPS 이상에서 이것이 중요한 이유를 설명할 수 있다.
- 데이터 거주성(data residency)과 운영 예산(ops budget)을 고려하여 자체 호스팅(self-hosted) vs 관리형(managed)을 선택할 수 있다.

## 문제 정의

귀사의 제품은 OpenAI, Anthropic, 그리고 자체 호스팅된 Llama를 호출합니다. 각 공급자는 서로 다른 SDK, 오류 모델, 요청 제한(rate limit), 인증 방식을 가지고 있습니다. 귀하는 다음과 같은 요구사항을 원합니다:
- 장애 조치(failover) (예: OpenAI에서 429 오류가 발생하면 Anthropic 시도)
- 단일 자격 증명 저장소
- 통합된 관측 가능성(observability)
- 테넌트별 요청 제한(rate limit)

애플리케이션 레이어에서 이를 재구현하면 모든 서비스가 각 공급자에 결합(couple)됩니다. 게이트웨이 레이어는 이를 하나의 프로세스(OpenAI 호환 API가 일반적)로 통합하여 여러 공급자로 분산(fan out)시킵니다.

## 개념

### 6가지 핵심 기능

1. **프로바이더 라우팅** — OpenAI, Anthropic, Gemini, 자체 호스팅 등을 하나의 API 뒤에 통합.
2. **폴백** — 429, 5xx 오류 또는 품질 실패 시 다른 곳에서 재시도.
3. **재시도** — 지수 백오프, 시도 횟수 제한.
4. **속도 제한** — 테넌트별, 키별, 모델별.
5. **시크릿 참조** — 런타임 시 볼트(vault)에서 자격 증명 가져오기 (앱에 저장하지 않음).
6. **관측 가능성** — OTel + GenAI 속성 (Phase 17 · 13) + 비용 추적.
7. **가드레일** — PII(개인 식별 정보) 삭제, 탈옥 감지, 허용 주제 필터.

### LiteLLM — MIT OSS, Python

- 100+ 프로바이더, OpenAI 호환, 라우터 구성, 폴백, 기본 관측 가능성.
- Kong 벤치마크에서 약 2000 RPS에서 성능 저하; 8GB 메모리 사용량, 지속적 부하 시 연쇄 실패 발생.
- 최적 사용처: Python 앱, <500 RPS, 개발/스테이징 게이트웨이, 실험적 라우팅.
- 비용: OSS는 $0; 클라우드 무료 티어 존재.

### Portkey — 제어 평면 포지셔닝

- 2026년 3월 기준 Apache 2.0 OSS. 가드레일, PII 삭제, 탈옥 감지, 감사 추적.
- 요청당 20-40ms 지연 오버헤드.
- 생산 티어(보존 + SLA) 월 $49.
- 최적 사용처: 가드레일 + 관측 가능성이 필요한 규제 산업.

### Kong AI 게이트웨이 — 확장성 중심

- Kong Gateway(루아+OpenResty 기반 성숙한 API 게이트웨이 제품) 위에 구축.
- Kong 자체 벤치마크(12-CPU 기준): Portkey보다 228% 빠름, LiteLLM보다 859% 빠름.
- 가격: 모델당 월 $100, Plus 티어 최대 5개.
- 최적 사용처: 이미 Kong 사용 중; >1000 RPS; 라이선스 구매 의향.

### Bifrost (Maxim AI)

- 구성 가능한 백오프로 자동 재시도.
- OpenAI 429 오류 시 Anthropic 폴백이 기본 레시피.
- 신규 진입자; 상용.

### Cloudflare AI 게이트웨이 / Vercel AI 게이트웨이

- 관리형, 운영 불필요. 기본 재시도 및 관측 가능성.
- 최적 사용처: Cloudflare/Vercel에서 엣지 서빙되는 JavaScript 앱.
- 가드레일 및 속도 제한 측면에서 Kong/Portkey에 비해 제한적.

### 자체 호스팅 vs 관리형

데이터 거주지가 강제 함수(forcing function). 의료 및 금융은 기본적으로 자체 호스팅(LiteLLM 또는 Portkey OSS 또는 Kong). 소비자 제품은 관리형(Cloudflare AI 게이트웨이) 또는 중간 계층(Portkey 관리형) 기본. 하이브리드: 규제 테넌트는 자체 호스팅, 나머지는 관리형.

### 지연 시간 예산

- LiteLLM: 일반적인 오버헤드 5-15ms.
- Portkey: 20-40ms 오버헤드.
- Kong: 3-8ms 오버헤드.
- Cloudflare/Vercel: 1-3ms 오버헤드(엣지 장점).

게이트웨이 지연 시간은 TTFT(첫 토큰 대기 시간)에 직접 추가. TTFT P99 < 100ms SLA에는 Kong 또는 Cloudflare. P99 < 500ms에는 모든 옵션 가능.

### 속도 제한 의미론 중요

단순 토큰 버킷은 중간 규모까지 작동. 멀티테넌트는 슬라이딩 윈도우 + 버스트 허용 + 테넌트별 계층화 필요. LiteLLM은 토큰 버킷 제공; Kong은 슬라이딩 윈도우 제공; Portkey는 계층화된 기능 제공.

### 게이트웨이 + 관측 가능성 + 라우팅 조합

Phase 17 · 13(관측 가능성) + 16(모델 라우팅) + 19(게이트웨이)는 프로덕션에서 동일한 계층. 세 가지를 모두 지원하는 도구 선택 또는 신중하게 연결: 2026년 대부분의 배포는 Helicone(관측 가능성) 또는 Portkey(가드레일)를 Kong(확장성)과 결합하여 역할 분담.

### 기억해야 할 숫자

- LiteLLM: ~2000 RPS에서 성능 저하, 8GB 메모리.
- Portkey: 20-40ms 오버헤드; 2026년 3월부터 Apache 2.0.
- Kong: Portkey보다 228% 빠름, LiteLLM보다 859% 빠름.
- Kong 가격: 모델당 월 $100, Plus 티어 최대 5개.
- Cloudflare/Vercel: 엣지에서 1-3ms 오버헤드.

## 사용 방법

`code/main.py`는 429/5xx 오류 주입 환경에서 3개 공급자 간 폴백(fallback)을 포함한 게이트웨이 라우팅을 시뮬레이션합니다. 지연 시간(latency), 재시도 비율(retry rate), 폴백 적중률(fallback hit rate)을 보고합니다.

## Ship It

이 레슨은 `outputs/skill-gateway-picker.md`를 생성합니다. 규모(scale), 운영 환경(ops posture), 규정 준수(compliance), 지연 시간 예산(latency budget)을 고려하여 게이트웨이를 선택합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. OpenAI→Anthropic→자체 호스팅 순으로 폴백(fallback)을 구성하세요. 공급자 오류율 5%에서 예상되는 히트율(hit rate)은 얼마인가요?
2. SLA가 300ms 기준선에서 TTFT(시간 대비 첫 토큰) P99 < 200ms입니다. 예산 범위 내에 있는 게이트웨이(gateway)는 무엇인가요?
3. 의료 고객사가 자체 호스팅(self-hosted) + PII(개인 식별 정보) 삭제 + 감사(audit)를 요구합니다. Portkey OSS와 Kong 중 어떤 것을 선택해야 하나요?
4. LiteLLM과 Kong을 비교하세요: 어떤 RPS(초당 요청 수) 임계값에서 팀이 마이그레이션해야 하나요?
5. 멀티테넌트 SaaS를 위한 속도 제한(rate-limit) 정책을 설계하세요: 무료 티어, 체험판 티어, 유료 티어. 토큰 버킷(token-bucket) 또는 슬라이딩 윈도우(sliding-window) 중 어떤 방식을 선택해야 하나요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Gateway | "API 브로커" | 애플리케이션과 공급자 사이에 위치하는 프로세스 |
| LiteLLM | "MIT에서 만든 것" | Python 오픈소스, 100+ 공급자 지원, 2K RPS에서 장애 발생 |
| Portkey | "가드레일 게이트웨이" | 제어 평면 + 관측 가능성, Apache 2.0 라이선스 |
| Kong AI Gateway | "확장성 좋은 것" | Kong Gateway 기반 구축, 벤치마크 성능 리더 |
| Bifrost | "Maxim의 게이트웨이" | 재시도 + Anthropic 폴백 레시피 지원 |
| Cloudflare AI Gateway | "엣지 관리형" | 엣지 배포 관리형 게이트웨이, 제로 운영 |
| PII redaction | "데이터 스크러빙" | 모델 전송 전 정규식 + NER 마스킹 |
| Jailbreak detection | "프롬프트 인젝션 가드" | 사용자 입력에 대한 분류기 |
| Audit trail | "규제 로그" | 모든 LLM 호출에 대한 불변 기록 |
| Token-bucket | "단순 속도 제한" | 리필 기반 속도 제한기 |
| Sliding-window | "정밀 속도 제한" | 시간 창 기반 속도 제한기; 더 나은 공정성 보장 |

## 추가 자료

- [Kong AI Gateway 벤치마크](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — 2026년 AI 게이트웨이 비교](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — 2026년 최고의 LLM 게이트웨이 도구](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway 문서](https://docs.konghq.com/gateway/latest/ai-gateway/)