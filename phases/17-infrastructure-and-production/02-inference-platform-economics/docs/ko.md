# 추론 플랫폼 경제학 — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> 2026년 추론 시장은 더 이상 GPU 시간 대여가 아닙니다. 맞춤형 실리콘(Groq, Cerebras, SambaNova), GPU 플랫폼(Baseten, Together, Fireworks, Modal), API 우선 마켓플레이스(Replicate, DeepInfra)로 분화됩니다. Fireworks는 2026년 5월 1일 GPU당 시간당 $1로 가격을 인상했으며, 하루 10T+ 토큰 처리량으로 $4B 기업 가치를 달성한 것은 볼륨 기반 모델의 성공을 보여줍니다. Baseten은 2026년 1월 $5B 기업 가치로 $300M 시리즈 E 펀딩을 마감했습니다. 경쟁 포지셔닝 규칙은 간단합니다: Fireworks는 지연 시간 최적화, Together는 카탈로그 범위 최적화, Baseten은 엔터프라이즈 완성도 최적화, Modal은 Python 네이티브 개발자 경험(DX) 최적화, Replicate는 멀티모달 범위 최적화, Anyscale은 분산 Python 최적화를 추구합니다. 이 레슨은 창업자에게 전달할 수 있는 매트릭스를 제공합니다.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 장난감 수준의 호출당 경제성 비교기)  
**선수 지식:** 17단계 · 01 (관리형 LLM 플랫폼), 17단계 · 04 (vLLM 서빙 내부 구조)  
**소요 시간:** ~60분

## 학습 목표

- 세 가지 시장 세그먼트(커스텀 실리콘, GPU 플랫폼, API 우선)를 나열하고 각 벤더를 해당 세그먼트에 매핑하시오.
- "토큰당(per-token)" API 가격 모델이 하드웨어가 아닌 서빙 엔진의 비용 곡선으로 압축되는 이유를 설명하시오.
- 최소 세 개 벤더에서 요청당 실효 비용을 계산하고, 토큰당(per-token)보다 분당(per-minute) 가격 모델(Baseten, Modal)이 더 유리한 경우를 설명하시오.
- 주어진 워크로드(서버리스 버스티, 안정적인 고처리량, 파인튜닝 변형, 멀티모달)에 적합한 기본 플랫폼을 식별하시오.

## 문제 정의

관리형 하이퍼스케일러 플랫폼을 평가했습니다. 더 좁고 빠른 공급자가 필요하다고 판단했습니다 — **Fireworks**는 지연 시간(latency), **Together**는 범위(breadth), **Baseten**은 미세 조정된 커스텀 모델(fine-tuned custom model)에 적합합니다. 이제 6개의 실제 선택지가 있지만, 가격 페이지가 일관되지 않습니다. **Fireworks**는 $/M 토큰, **Baseten**은 $/분, **Modal**은 $/초, **Replicate**는 $/예측(prediction) 단위로 가격을 표시합니다. 워크로드를 모델링하지 않고는 이들을 직접 비교할 수 없습니다.

더 큰 문제는 각 가격 페이지의 비즈니스 모델이 다르다는 점입니다. **Fireworks**는 공유 GPU에서 자체 커스텀 엔진(FireAttention)을 실행하며, 토큰당 요금은 활용률 곡선을 반영합니다. **Baseten**은 Truss + 전용 GPU를 제공하며, 분당 요금은 독점성(exclusivity)을 반영합니다. **Modal**은 진정한 Python 서버리스(Serverless)로, 초당 과금에 초 단위 미만의 콜드 스타트(cold start)가 발생합니다. 동일한 출력(LLM 응답)이지만, 세 가지 다른 비용 함수가 존재합니다.

이 강의에서는 6개 플랫폼을 모델링하고 각 플랫폼이 언제 가장 유리한지 설명합니다.

## 개념

### 세 가지 세그먼트

**커스텀 실리콘** — Groq (LPU), Cerebras (WSE), SambaNova (RDU). 일반적으로 동일한 모델에서 GPU 기반 클러스터 대비 5-10배 빠른 디코딩 성능을 제공합니다. 토큰당 가격은 더 높지만(Groq는 2025년 말 기준 Llama-70B에서 ~$0.99/M), 지연 시간에 민감한 사용 사례에서는 타의 추종을 불허합니다. Groq는 음성 에이전트 및 실시간 번역 분야에서 프로덕션 선택지로 선호됩니다.

**GPU 플랫폼** — Baseten, Together, Fireworks, Modal, Anyscale. NVIDIA (2026년 H100, H200, B200) 또는 때때로 AMD에서 실행됩니다. "RAW GPU 임대"(RunPod, Lambda)와 "하이퍼스케일러 관리 서비스"(Bedrock) 사이의 경제적 계층입니다.

**API 우선 마켓플레이스** — Replicate, DeepInfra, OpenRouter, Fal. 광범위한 카탈로그, 예측당 과금 또는 초당 과금, 첫 호출 시간 최소화를 강조합니다.

### Fireworks — 지연 시간 최적화 GPU 플랫폼

- FireAttention 엔진(커스텀); 동등한 구성에서 vLLM 대비 4배 낮은 지연 시간을 마케팅합니다.
- 비대화형 워크로드의 경우 서버리스 요금의 ~50% 배치 티어.
- 파인튜닝된 모델을 베이스 모델과 동일한 요금으로 제공 — LoRA에 프리미엄을 부과하는 공급업체와 차별화되는 진정한 강점.
- 2026년 중반: 2026년 5월 1일 기준 온디맨드 GPU 임대 $1/시간 도입. 대량 구매 시 가격 협상 가능.
- 재무 신호: $4B 기업 가치, 일일 10T+ 토큰 처리.

### Together — 폭 최적화

- 오픈소스 모델 출시 후 며칠 이내에 200+ 모델 제공.
- 동등한 LLM 모델에서 Replicate 대비 50-70% 저렴 — "AI 네이티브 클라우드" 포지셔닝은 볼륨과 카탈로그에 중점.
- 추론 + 파인튜닝 + 학습을 단일 API로 통합.

### Baseten — 엔터프라이즈 최적화

- Truss 프레임워크: 종속성, 시크릿, 서빙 구성을 하나의 매니페스트로 패키징.
- T4부터 B200까지 GPU 범위. 분 단위 과금 및 합리적인 콜드 스타트 완화.
- SOC 2 Type II, HIPAA 준비 완료. 일반적인 핀테크 및 헬스케어 선택.
- 2026년 1월 시리즈 E($300M, CapitalG, IVP, NVIDIA 투자)에서 $5B 기업 가치.

### Modal — Python 네이티브 최적화

- 순수 Python으로 인프라 코드화. `@modal.function(gpu="A100")` 데코레이터로 함수를 장식하고 한 번의 명령으로 배포.
- 초당 과금. 사전 예열 시 콜드 스타트 2-4초, 소형 모델은 <1초.
- 2025년 시리즈 B에서 $87M 투자 유치, $1.1B 기업 가치. 독립 설문조사에서 가장 강력한 개발자 경험 점수.

### Replicate — 멀티모달 폭

- 예측당 과금. 이미지, 비디오, 오디오 모델의 기본 플랫폼.
- 통합 생태계(Zapier, Vercel, CMS 플러그인).
- LLM 토큰당 요금은 경쟁력이 떨어지지만 멀티모달 다양성에서 승리.

### Anyscale — Ray 네이티브

- Ray 기반; RayTurbo는 Anyscale의 독점 추론 엔진(vLLM과 경쟁).
- 추론 단계가 더 큰 그래프의 한 노드인 분산 Python 워크로드에 최적.
- 관리형 Ray 클러스터; Ray AIR 및 Ray Serve와 긴밀한 통합.

### 토큰당 과금 vs 분 단위 과금 — 각각의 우위

토큰당 과금은 워크로드가 지연 시간에 민감하지 않고 버스트형일 때 적합합니다. 사용한 만큼만 지불합니다. 분 단위 과금은 GPU 사용률이 높고 예측 가능할 때 유리합니다. GPU를 포화 상태로 사용하면 토큰당 과금을 능가합니다.

대략적 규칙: 전용 GPU의 지속적 사용률이 ~30% 이상인 워크로드의 경우 분 단위 과금(Baseten, Modal)이 토큰당 과금(Fireworks, Together)을 능가하기 시작합니다. 그 미만에서는 유휴 시간에 대한 비용을 피할 수 있어 토큰당 과금이 유리합니다.

### 커스텀 엔진이 진정한 해자

vLLM과 SGLang 이상의 모든 플랫폼은 커스텀 엔진을 주장합니다. FireAttention, RayTurbo, Baseten의 추론 스택. 커스텀 엔진 주장은 마케팅에 가깝지만, 솔직한 프레임은 vLLM + SGLang이 오픈소스 추론의 약 80%를 차지하며, 플랫폼 계층의 차별점은 DX(개발자 경험), 어트리뷰션, SLA입니다.

### 기억해야 할 숫자

- Fireworks GPU 임대: 2026년 5월 1일 기준 $1/시간 인상.
- Fireworks 주장: 동등한 구성에서 vLLM 대비 4배 낮은 지연 시간.
- Together: LLM에서 Replicate 대비 50-70% 저렴.
- Baseten 기업 가치: $5B(2026년 1월 시리즈 E, $300M 라운드).
- Modal 기업 가치: $1.1B(2025년 시리즈 B).
- 분 단위 과금은 지속적 사용률 ~30% 이상에서 토큰당 과금을 능가.

## 사용 방법

`code/main.py`는 합성 워크로드에서 6개 공급업체를 가격 모델별로 비교합니다. 일일 비용($/day)과 유효 토큰당 비용($/M tokens)을 보고합니다. 이 스크립트를 실행하여 토큰당 과분당 가격 모델 간의 손익분기점을 확인할 수 있습니다.

## Ship It

이 레슨은 `outputs/skill-inference-platform-picker.md`를 생성합니다. 워크로드 프로파일, SLA(서비스 수준 계약), 예산을 고려하여 주요 추론 플랫폼을 선택하고, 차순위 후보를 명시합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 70B 모델에서 Baseten(분 단위)이 Fireworks(토큰 단위)를 지속적으로 능가하는 사용률(utilization)은 얼마인가요? 교차점(crossover)을 직접 계산하고, 경험 법칙(rule of thumb)과 비교하세요.
2. 귀사의 제품은 이미지 생성, 채팅, 음성-텍스트 변환을 제공합니다. 각 모달리티에 적합한 플랫폼을 선택하고, 이를 통합하는 게이트웨이 패턴(gateway pattern)을 명시하세요.
3. Fireworks가 주요 모델 가격을 시간당 $1 인상합니다. 트래픽의 40%가 배치 티어(50% 할인)로 이동할 때 혼합 비용(blended cost) 영향을 모델링하세요.
4. 규제 대상 고객이 SOC 2 Type II + HIPAA + 전용 GPU를 요구합니다. 어떤 세 가지 플랫폼이 적합하며, FinOps 관점에서 어떤 플랫폼이 가장 우수한가요?
5. Fireworks 서버리스, Together 온디맨드, Baseten 전용, Replicate API에서 Llama 3.1 70B의 1,000회 예측당 비용을 비교하세요. 하루 10회 예측 시 가장 저렴한 옵션은? 하루 10,000회 예측 시에는?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 커스텀 실리콘(Custom silicon) | "비-GPU 칩" | Groq LPU, Cerebras WSE, SambaNova RDU — 디코딩 최적화 |
| 파이어어텐션(FireAttention) | "불꽃놀이 엔진" | 커스텀 어텐션 커널; vLLM 대비 4배 낮은 지연 시간 마케팅 |
| 트러스(Truss) | "Baseten의 포맷" | 모델 패키징 매니페스트; 의존성 + 시크릿 + 서빙 설정 |
| 퍼토큰(Per-token) | "API 가격 책정" | 소비된 토큰 단위 과금; 유휴 시간 비용 없음 |
| 퍼미닛(Per-minute) | "전용 가격 책정" | 벽시계 GPU 시간 단위 과금; 고부하 시 유리 |
| 퍼프레딕션(Per-prediction) | "Replicate 가격 책정" | 모델 호출당 과금; 이미지/비디오에 일반적 |
| 레이터보(RayTurbo) | "Anyscale 엔진" | Ray 기반 전용 추론; Ray 클러스터에서 vLLM과 경쟁 |
| 배치 티어(Batch tier) | "50% 할인" | 비대화형 큐 할인 요금; Fireworks, OpenAI에서 일반적 |
| 베이스 레이트 파인튜닝(Fine-tuned at base rate) | "Fireworks LoRA" | LoRA 서빙 요청을 베이스 모델 요금으로 과금 (차별화 요소)

## 추가 자료

- [Fireworks Pricing](https://fireworks.ai/pricing) — 토큰당 요금, 배치 티어, GPU 임대.
- [Baseten Pricing](https://www.baseten.co/pricing/) — 분당 요금, 약정 용량, 엔터프라이즈 티어.
- [Modal Pricing](https://modal.com/pricing) — 초당 GPU 요금 및 무료 티어.
- [Together AI Pricing](https://www.together.ai/pricing) — 모델 카탈로그 및 토큰당 요금.
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo 및 관리형 Ray 요금.
- [Northflank — Fireworks AI 대안](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 비교 평가.
- [Infrabase — 2026년 AI 추론 API 제공업체](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 벤더 환경.