# 관리형 LLM 플랫폼 — Bedrock, Vertex AI, Azure OpenAI

> 세 가지 하이퍼스케일러, 세 가지 전략. AWS Bedrock은 모델 마켓플레이스 — Claude, Llama, Titan, Stability, Cohere를 하나의 API로 통합. Azure OpenAI는 독점적인 OpenAI 파트너십과 전용 용량을 위한 Provisioned Throughput Units(PTUs)를 제공. Vertex AI는 Gemini 중심으로 최고의 장문 컨텍스트 및 멀티모달 기능을 갖춤. 2026년 Artificial Analysis에 따르면 Azure OpenAI는 ~50ms, Bedrock은 Llama 3.1 405B 동급 모델에서 ~75ms의 중앙값 응답 시간을 기록 — PTUs는 전용 용량이 공유 온디맨드보다 우수함을 설명. 결정 기준은 "어떤 것이 가장 빠른가"가 아니라 "어떤 모델 카탈로그와 FinOps(클라우드 재무 운영) 기능이 내 제품과 맞는가"입니다. 이 레슨은 직감이 아닌 트레이드오프를 문서화하여 선택하는 방법을 가르칩니다.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 간단한 비용-지연 시간 비교기)  
**선수 조건:** 11단계(LLM 엔지니어링), 13단계(도구 및 프로토콜)  
**소요 시간:** ~60분

## 학습 목표

- 세 가지 플랫폼 전략(마켓플레이스 vs 독점 vs 제미니-퍼스트)을 명명하고 각각을 제품 사용 사례에 매칭할 수 있다.
- Azure OpenAI에서 Provisioned Throughput Units(PTU, 프로비저닝 처리량 단위)가 제공하는 이점과, 405B 규모에서 온디맨드 Bedrock이 일반적으로 ~25ms 더 느리게 읽는 이유를 설명할 수 있다.
- 각 플랫폼(Bedrock 애플리케이션 추론 프로파일 vs Vertex 팀별 프로젝트 vs Azure 범위 + PTU 예약)에 대한 FinOps(클라우드 재무 운영) 귀속 범위를 도식화할 수 있다.
- "최소 두 공급자" 정책을 작성하고, 단일 벤더 종속성이 2026년에 비용이 많이 드는 실수가 되는 이유를 설명할 수 있다.

## 문제 정의

Claude 3.7 Sonnet을 제품에 선택했습니다. 이제 이를 서빙해야 합니다. Anthropic API를 직접 호출하거나, AWS Bedrock을 통해 호출하거나, 게이트웨이를 경유할 수 있습니다. 직접 API 호출이 가장 간단합니다. Bedrock은 BAA(기업협약), VPC 엔드포인트, IAM, CloudWatch 어트리뷰션을 추가합니다. 게이트웨이는 장애 조치(failover), 통합 청구, 공급자 간 속도 제한을 제공합니다.

더 깊은 문제는 카탈로그입니다. 동일한 제품에서 Claude, Llama, Gemini이 모두 필요한 경우, Bedrock, Vertex, Azure OpenAI를 동시에 사용하지 않는 한 한 곳에서 모두 구매할 수 없습니다. 초대형 클라우드 제공업체(hyperscaler)들은 상호 교환 불가능합니다. 각 업체는 모델 계층 소유권을 놓고 서로 다른 전략을 선택했기 때문입니다.

이 강의에서는 세 가지 전략, 지연 시간 격차, FinOps 격차, 종속성 위험을 매핑합니다.

## 개념

### 세 가지 전략

**AWS Bedrock** — 마켓플레이스. Claude(Anthropic), Llama(Meta), Titan(AWS 자체 모델), Stability(이미지), Cohere(임베딩), Mistral, 이미지 및 임베딩 하위 카탈로그 포함. 하나의 API, 하나의 IAM 인터페이스, 하나의 CloudWatch 내보내기. Bedrock의 전략은 고객이 단일 모델보다 옵션 다양성을 더 원한다는 것입니다.

**Azure OpenAI** — 독점 파트너십. Azure 데이터센터에서 GPT-4/4o/5/o-시리즈, DALL·E, Whisper 및 OpenAI 모델 파인튜닝(fine-tuning) 제공. "Azure OpenAI 서비스" 카탈로그에는 OpenAI 외 모델이 없음 — 다른 모델은 Azure AI Foundry(별도 제품)로 이동. Azure의 전략은 OpenAI가 최전선 모델이며 고객이 해당 관계에 대한 엔터프라이즈 제어를 원한다는 것입니다.

**Vertex AI** — Gemini 우선, 나머지는 후순위. Gemini 1.5/2.0/2.5 Flash 및 Pro, Model Garden(서드파티) 제공. Vertex의 전략은 멀티모달 장기 컨텍스트 — 1M 토큰 Gemini 컨텍스트가 차별화 요소입니다.

### 대규모에서의 지연 시간 격차

Artificial Analysis는 지속적인 벤치마크를 실행합니다. 동등한 Llama 3.1 405B 배포(공유 온디맨드)에서 Azure OpenAI의 중앙값 첫 토큰 지연 시간(TTFT)은 약 50ms, Bedrock은 약 75ms입니다. 이 격차는 AWS 실패가 아닌 용량 모델 차이 때문입니다. Azure는 PTU(Provisioned Throughput Units)를 판매하며, 이는 테넌트를 위한 GPU 용량을 예약합니다. Bedrock의 동등한 기능(Provisioned Throughput)도 있지만 시간당 단위당 약 $21부터 시작하며, 대부분의 고객은 공유 온디맨드에 머뭅니다.

온디맨드 공유 용량은 다른 모든 고객의 트래픽과 경쟁합니다. 전용 용량은 그렇지 않습니다. 제품 SLA가 P99에서 TTFT < 100ms라면, Azure에서 PTU를 구매하거나 Bedrock Provisioned Throughput을 구매하거나 기본 변동성을 수용해야 합니다.

### Provisioned Throughput 경제학

**Azure PTU**: 추론 컴퓨팅의 예약된 블록. 예측 가능한 워크로드의 경우 온디맨드 대비 최대 ~70% 절감. 트래픽과 무관하게 시간당 고정 비용 — 유휴 시에도 예약 비용 지불. 손익분기점은 일반적으로 40-60% 지속 활용도입니다.

**Bedrock Provisioned Throughput**: 모델 및 리전에 따라 시간당 $21-$50. 유사한 계산 — 손익분기점은 피크 활용도의 약 50%. 월간 약정 필요.

**Vertex** 전용 용량은 Gemini SKU별로 판매되며, 가격은 모델 및 리전에 따라 다르고 공개적으로 덜 광고됩니다.

### FinOps 표면 — 진정한 차별화 요소

**Bedrock 애플리케이션 추론 프로필**은 마켓플레이스에서 가장 깔끔한 비용 귀속입니다. 프로필에 `팀`, `제품`, `기능` 태그 추가 → 모든 모델 호출을 라우팅 → CloudWatch가 후처리 없이 프로필별 비용을 분류합니다. 2025년 추가되었으며, 여전히 가장 세분화된 하이퍼스케일러 네이티브 기능입니다.

**Vertex** 귀속은 프로젝트별 팀 + 라벨-모든 곳에. 각 팀을 GCP 프로젝트로 모델링하고 모든 리소스에 라벨을 추가한 후 BigQuery 청구 내보내기 + DataStudio로 집계. 더 많은 작업이 필요하지만 BigQuery를 통해 비용 데이터에 임의의 SQL을 적용할 수 있습니다.

**Azure**는 구독/리소스 그룹 범위 + 태그에 의존하며, PTU 예약은 1급 비용 객체입니다. 태그는 요청이 아닌 리소스 그룹에서 상속되므로, 요청별 귀속에는 Application Insights 사용자 지정 메트릭 또는 헤더를 스탬핑하는 게이트웨이가 필요합니다.

패턴: Bedrock은 가장 깔끔한 네이티브, Vertex는 BigQuery를 통한 최대 유연성, Azure는 계측하지 않으면 가장 불투명합니다.

### 잠금(Lock-in)은 2026년 리스크

단일 하이퍼스케일러 커밋은 하나의 모델이 지배적일 때 괜찮았습니다. 2026년에는 최전선이 매월 이동합니다 — 한 분기에는 Claude 3.7, 다음 분기에는 Gemini 2.5, 그 다음 분기에는 GPT-5. 하나의 플랫폼에 잠그면 최전선의 2/3에서 제외됩니다.

작동하는 팀의 패턴: 제품-중요 LLM 호출에 최소 2개 공급자 채택. 일반적인 조합은 Bedrock + Azure OpenAI — Claude는 하나, GPT는 다른 쪽에서, 상호 장애 조치, 동일한 게이트웨이. 게이트웨이가 최적을 라우팅하므로 비용 증가는 미미하지만, 장애 발생 시(예: 2025년 1월 Azure OpenAI 사고, AWS us-east-1 장애) 가용성 향상은 결정적입니다.

### 데이터 거주성, BAA, 규제 산업

**Bedrock**: 대부분의 리전에서 BAA, VPC 엔드포인트, 가드레일. 일반적인 핀테크 기본값.
**Azure OpenAI**: HIPAA, SOC 2, ISO 27001; EU 데이터 거주성; 엔터프라이즈-규제 기본값.
**Vertex**: HIPAA, GDPR, 리전별 데이터 거주성; Google Cloud의 규정 준수 스택.

세 가지 모두 기본 체크박스를 충족합니다. 차이는 데이터 보존 정책, 로그 처리 방식, 악용 모니터링이 트래픽을 읽는지 여부(대부분 기본 옵트인; 엔터프라이즈는 옵트아웃 가능)에 있습니다.

### 기억해야 할 숫자

- Azure OpenAI의 Llama 3.1 405B 동등 모델 중앙값 TTFT(PTU 사용): ~50ms.
- Bedrock 온디맨드 중앙값 TTFT: ~75ms.
- Bedrock Provisioned Throughput: 시간당 $21-$50/단위.
- Azure PTU 손익분기점: ~40-60% 지속 활용도.
- 높은 활용도에서의 PTU 절감(온디맨드 대비): 최대 70%.

## 사용 방법

`code/main.py`는 합성 워크로드에서 세 플랫폼을 비교합니다 — 온디맨드(On-Demand) 대 PTU(Provisioned Throughput Units) 경제성, TTFT(Time To First Token) 분산, 비용 할당 정확도를 모델링합니다. 이를 실행하여 PTU가 수익을 내는 지점과 마켓플레이스의 모델 다양성이 TTFT 격차를 상쇄하는 지점을 확인하세요.

## Ship It

이 레슨은 `outputs/skill-managed-platform-picker.md`를 생성합니다. 워크로드 프로필(필요한 모델, TTFT SLA, 일일 처리량, 규정 준수 요구사항)을 기반으로 주요 플랫폼, 대체 플랫폼, FinOps 계측 계획을 추천합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 70B 클래스 모델에서 Azure PTU가 온디맨드를 지속적으로 능가하는 사용률(utilization)은 얼마인가요? 손익분기점(break-even)을 계산하고, 광고된 40-60% 범위와 비교하세요.  
2. 귀사의 제품에는 Claude 3.7 Sonnet과 GPT-4o가 필요합니다. 두 공급자(provider) 배포를 설계하세요 — 각 모델은 어떤 하이퍼스케일러(hyperscaler)에 배치할지, 프런트엔드에 어떤 게이트웨이(gateway)를 둘지, 장애 조치(failover) 정책은 어떻게 구성할지 설명하세요.  
3. 규제를 받는 의료 고객이 BAA(사업 협약), 미국 동부 데이터 레지던시(US-East data residency), P99 TTFT(첫 토큰 대기 시간) 100ms 미만을 요구합니다. 플랫폼을 선택하고 다음 세 가지 구체적인 기능으로 정당화하세요.  
4. Bedrock 요금이 트래픽 변화 없이 이번 달에 4배 증가한 것을 발견했습니다. 애플리케이션 추론 프로파일(Application Inference Profiles)이 없을 때, 원인을 어떻게 찾을 수 있나요? 프로파일을 사용하면 문제 해결에 얼마나 걸리나요?  
5. Azure OpenAI와 Bedrock 가격 페이지를 읽으세요. 월 100M 토큰 Claude 워크로드의 경우, 다음 중 어떤 옵션이 더 저렴한가요? — 직접 Anthropic API, Bedrock 온디맨드, Bedrock 프로비저닝 처리량(Provisioned Throughput)  

> **참고**: 번역 시 다음 규칙을 적용했습니다.  
> - 전문 용어는 한국어(영어) 형식으로 표기 (예: 손익분기점(break-even), P99 TTFT(P99 TTFT))  
> - 코드 블록, 수식, URL, 라이브러리/프레임워크 고유명은 번역하지 않음  
> - 마크다운 구조(헤딩, 목록, 강조 등)는 원본과 동일하게 유지

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Bedrock | "AWS LLM 서비스" | Claude, Llama, Titan, Mistral, Cohere를 아우르는 모델 마켓플레이스 |
| Azure OpenAI | "Azure의 ChatGPT" | 엔터프라이즈 제어 기능이 있는 Azure 데이터센터 내 독점 OpenAI 모델 |
| Vertex AI | "Google의 LLM" | 제3자 모델을 위한 Model Garden을 갖춘 Gemini 우선 플랫폼 |
| PTU | "전용 용량" | 프로비저닝된 처리량 단위(Provisioned Throughput Unit) — 시간당 요금제로 예약된 추론 GPU |
| Application Inference Profile | "Bedrock 태깅" | 태그, CloudWatch 네이티브 기능이 포함된 제품별 비용/사용량 프로필 |
| Model Garden | "Vertex 카탈로그" | Gemini와 별도로 구성된 Vertex AI의 제3자 모델 섹션 |
| Two-provider minimum | "LLM 중복성" | 모든 중요한 LLM 경로를 ≥2개의 하이퍼스케일러에서 실행하는 정책 |
| BAA | "HIPAA 서류 작업" | 비즈니스 협력 계약(Business Associate Agreement); PHI(Protected Health Information) 필수 요건; 3사 모두 제공 |
| Abuse monitoring | "로그 감시자" | 프롬프트/출력에 대한 공급자 측 안전 검사; 엔터프라이즈에서 선택적 제외 가능 |

## 추가 자료

- [AWS Bedrock 가격 책정](https://aws.amazon.com/bedrock/pricing/) — 공식 요금표 및 프로비저닝 처리량(Provisioned Throughput) 가격.
- [Azure OpenAI 서비스 가격 책정](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU(Provisioned Throughput Unit) 경제 모델 및 요금표.
- [Vertex AI 생성형 AI 가격 책정](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 모델 계층 및 Model Garden 추가 요금.
- [Artificial Analysis LLM 리더보드](https://artificialanalysis.ai/) — 공급자별 지속적인 지연 시간 및 처리량 벤치마크.
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO 가이드 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 엔터프라이즈 의사 결정 프레임워크.
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 비용 할당 메커니즘 비교.