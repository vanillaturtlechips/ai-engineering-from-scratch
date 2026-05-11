# 모더레이션 시스템 — OpenAI, Perspective, Llama Guard

> 프로덕션 모더레이션 시스템은 레슨 12-16에서 정의된 안전 정책을 운영화합니다. OpenAI 모더레이션 API: `omni-moderation-latest`(2024)는 GPT-4o 기반으로 텍스트와 이미지를 한 번의 호출로 분류합니다. 이전 버전 대비 다국어 테스트 세트에서 42% 더 우수하며, 응답 스키마는 13개 범주 부울을 반환합니다 — 괴롭힘, 괴롭힘/위협, 증오, 증오/위협, 불법, 불법/폭력, 자해, 자해/의도, 자해/지침, 성적, 성적/미성년자, 폭력, 폭력/그래픽. 대부분의 개발자에게 무료입니다. 계층적 패턴: 입력 모더레이션(생성 전), 출력 모더레이션(생성 후), 커스텀 모더레이션(도메인 규칙). 비동기 병렬 호출로 지연 시간 숨김; 플래그 시 플레이스홀더 응답. Llama Guard 3/4(레슨 16): 14개 MLCommons 위험 요소, 코드 인터프리터 남용, 8개 언어(v3), 다중 이미지(v4). Perspective API(Google Jigsaw): LLM-as-moderator 시대 이전의 독성 점수 평가; 주로 단일 차원 독성(심각한 독성/모욕/욕설 변형 포함); 콘텐츠 모더레이션 연구 기준선. 폐기: Azure Content Moderator는 2024년 2월 폐기, 2027년 2월 퇴역, Azure AI Content Safety로 대체.

**유형:** 구축  
**언어:** Python(표준 라이브러리, 3계층 모더레이션 하니스)  
**사전 요구 사항:** 18단계 · 16(Llama Guard / Garak / PyRIT)  
**소요 시간:** ~60분

## 학습 목표

- OpenAI Moderation API의 카테고리 분류 체계와 Llama Guard 3의 MLCommons 집합과의 차이점을 설명하세요.
- 세 가지 모더레이션 계층 패턴(입력, 출력, 커스텀)을 설명하고 각각의 실패 사례 하나를 제시하세요.
- Perspective API의 LLM 시대 이전 기준선 역할과 연구 분야에서 계속 사용되는 이유를 설명하세요.
- Azure의 지원 종료 일정을 명시하세요.

## 문제 정의

레슨 12-16은 공격 및 방어 도구링을 설명합니다. 레슨 29는 사용자가 제품과 상호작용하는 표면 수준에서 방어를 운영화하는 배포된 조정 시스템을 다룹니다. 3계층 패턴은 2026년 기본 구성입니다.

## 개념

### OpenAI Moderation API

`omni-moderation-latest` (2024). GPT-4o 기반. 텍스트 + 이미지를 한 번의 호출로 분류. 대부분의 개발자에게 무료.

카테고리 (응답 스키마의 13개 부울):
- 괴롭힘, 괴롭힘/위협
- 증오, 증오/위협
- 자해, 자해/의도, 자해/지침
- 성적, 성적/미성년자
- 폭력, 폭력/그래픽
- 불법, 불법/폭력

멀티모달 지원은 `폭력`, `자해`, `성적`에 적용되지만 `성적/미성년자`에는 적용되지 않음; 나머지는 텍스트 전용.

`code/main.py`의 코드 하네스에서는 교육적 단순화를 위해 `/위협`, `/의도`, `/지침`, `/그래픽` 하위 카테고리를 최상위 부모로 통합. 프로덕션 코드는 전체 13개 카테고리 스키마를 사용해야 함.

다국어 테스트 세트에서 이전 세대 모더레이션 엔드포인트보다 42% 더 우수. 카테고리별 점수; 애플리케이션은 임계값을 설정.

### Llama Guard 3/4

레슨 16에서 다룸. 14개의 MLCommons 위험 카테고리 (OpenAI의 13개 응답 스키마 부울과 다르게 구성). 8개 언어 지원 (v3). Llama Guard 4 (2025년 4월)는 네이티브 멀티모달, 12B.

OpenAI와 Llama Guard의 분류 체계는 겹치지만 차이가 있음. OpenAI는 "불법"을 광범위한 카테고리로 분류; Llama Guard는 "폭력 범죄"와 "비폭력 범죄"를 별도로 분류. 배포는 정책-분류 체계 적합성에 따라 선택.

### Perspective API (Google Jigsaw)

LLM 기반 모더레이션 이전(2020년 이전)의 독성 점수 시스템. 카테고리: TOXICITY, SEVERE_TOXICITY, INSULT, PROFANITY, THREAT, IDENTITY_ATTACK. 단일 차원 기본 점수(TOXICITY)와 하위 차원 변형.

API가 안정적이고 문서화되어 있으며 수년간의 보정 데이터가 있어 콘텐츠 모더레이션 연구 기준으로 널리 사용. 현대적인 LLM 관련 사용 사례에는 일반적으로 Llama Guard 또는 OpenAI Moderation이 더 적합.

### 3계층 패턴

1. **입력 모더레이션.** 생성 전 사용자 프롬프트를 분류. 플래그 시 거부. 지연 시간: 분류기 1회 호출.
2. **출력 모더레이션.** 전달 전 모델 출력을 분류. 플래그 시 거부로 대체. 지연 시간: 생성 후 분류기 1회 호출.
3. **커스텀 모더레이션.** 도메인별 규칙(정규식, 허용 목록, 비즈니스 정책). 입력 또는 출력에서 실행.

3계층은 설계상 순차적: 입력 모더레이션은 생성 전에 완료되어야 하며, 출력 모더레이션은 생성 후에 실행. 병렬성은 계층 내에서 적용 — 동일한 텍스트에 대해 여러 분류기(예: OpenAI Moderation + Llama Guard + Perspective)를 동시에 실행하면 분류기별 지연 시간이 숨겨짐. 선택적 최적화로, 입력 모더레이션 완료 및 토큰-1 스트리밍 지연 중에 플레이스홀더 응답("잠시 확인 중...")을 표시할 수 있음. 플래그 동작은 구성 가능: 거부, 정제, 인간 검토로 에스컬레이션.

### 실패 모드

- **입력 전용.** 출력 환각(레슨 12-14의 인코딩 공격은 입력 분류기를 우회)을 잡지 못함.
- **출력 전용.** 모든 입력이 모델에 도달; 비용 증가; 공격자에게 내부 추론 노출.
- **커스텀 전용.** 카테고리 전반에 걸쳐 견고하지 않음; 정규식은 취약.

계층화가 기본. 이중 안전장치.

### Azure 사용 중단

Azure Content Moderator: 2024년 2월 사용 중단, 2027년 2월 폐기. Azure AI Content Safety로 대체, LLM 기반이며 Azure OpenAI와 통합. 마이그레이션은 Azure 배포를 위한 2024-2027 필드 수준 프로젝트.

### Phase 18에서의 위치

레슨 16은 레드 팀 컨텍스트에서 모더레이션 도구를 다룸. 레슨 29는 배포된 모더레이션을 다룸. 레슨 30은 현재의 이중 사용 능력 증거로 마무리.

## 사용 방법

`code/main.py`는 3계층 모더레이션 시스템을 구축합니다: 입력 모더레이터(키워드 + 카테고리 점수), 출력 모더레이터(출력에 동일한 분류기 적용), 커스텀 모더레이터(도메인 규칙). 입력 데이터를 실행하여 각 계층에서 어떤 내용을 감지하는지 관찰할 수 있습니다.

## Ship It

이 레슨은 `outputs/skill-moderation-stack.md`를 생성합니다. 배포(deployment)가 주어졌을 때, 입력 시 적용할 분류기(classifier), 출력 시 적용할 분류기, 사용자 정의 규칙(custom rules), 그리고 엣지 케이스(edge cases) 판단을 위한 심사관(judge) 등 모더레이션 스택(moderation stack) 구성을 권장합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 모든 세 개의 레이어에 대해 양성(benign), 경계성(borderline), 유해(harmful) 입력을 각각 실행하세요. 각 입력에 대해 어떤 레이어가 반응(fire)하는지 보고하세요.

2. 특정 범주에 대한 Perspective-API 스타일의 독성(toxicity) 점수 매기기를 하니스에 추가하세요. 해당 범주의 임계값(threshold) 동작을 범주 점수와 비교하세요.

3. OpenAI Moderation API 문서와 Llama Guard 3 범주 목록을 읽으세요. 각 OpenAI 범주를 가장 가까운 Llama Guard 범주에 매핑하세요. 깔끔하게 매핑되지 않는 세 가지 범주를 식별하세요.

4. 코드 어시스턴트 배포(예: GitHub Copilot)를 위한 조정 스택을 설계하세요. 가장 관련성이 높은 범주와 가장 관련성이 낮은 범주를 식별하고 사용자 정의 규칙을 제안하세요.

5. Azure Content Moderator는 2027년 2월에 지원 종료됩니다. Azure AI Content Safety로의 마이그레이션 계획을 수립하세요. 마이그레이션에서 가장 높은 위험 요소를 식별하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| OpenAI Moderation | "omni-moderation-latest" | GPT-4o 기반 13개 범주(텍스트) 분류기(부분적 멀티모달 지원) |
| Perspective API | "Google Jigsaw toxicity" | LLM 시대 이전 독성 점수 기준선 |
| Llama Guard | "MLCommons 14-category" | Meta의 위험 분류기(v3: 8B 텍스트, 8개 언어; v4: 12B 멀티모달) |
| 입력 모더레이션 | "pre-generation filter" | 모델 호출 전 사용자 프롬프트에 대한 분류기 |
| 출력 모더레이션 | "post-generation filter" | 전달 전 모델 출력에 대한 분류기 |
| 커스텀 모더레이션 | "domain rules" | 배포별 규칙(정규식, 허용 목록, 정책) |
| 계층적 모더레이션 | "all three layers" | 표준 프로덕션 배포 패턴 |

## 추가 자료

- [OpenAI Moderation API 문서](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation 엔드포인트
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard 저장소
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — 유해성 점수 평가
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — Azure 대체 서비스