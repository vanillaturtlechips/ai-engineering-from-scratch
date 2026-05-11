# 캡스톤 06 — 쿠버네티스용 DevOps 문제 해결 에이전트

> AWS의 DevOps 에이전트가 GA(일반 공급) 단계에 도달했고, Resolve AI가 K8s 플레이북을 공개했으며, NeuBird가 시맨틱 모니터링을 시연했고, Metoro는 AI SRE(사이트 신뢰성 엔지니어링)를 서비스별 SLO(서비스 수준 목표)에 연결했습니다. 프로덕션 환경은 다음과 같이 정착되었습니다: 알림 웹훅이 실행되고, 에이전트가 텔레메트리를 읽으며, K8s 객체 그래프를 탐색하고, 근본 원인 가설을 순위 매긴 후 승인 버튼이 포함된 Slack 요약 정보를 게시합니다. 기본적으로 읽기 전용이며, 모든 복구는 인간의 승인을 받아야 합니다. 이 캡스톤은 해당 에이전트를 구현하며, 20개의 합성 인시던트를 기반으로 평가되며, AWS 에이전트와의 3가지 공유 사례를 비교합니다.

**유형:** 캡스톤  
**언어:** Python (에이전트), TypeScript (Slack 통합)  
**사전 요구 사항:** 11단계(LLM 엔지니어링), 13단계(도구 및 MCP), 14단계(에이전트), 15단계(자율성), 17단계(인프라), 18단계(안전성)  
**적용 단계:** P11 · P13 · P14 · P15 · P17 · P18  
**소요 시간:** 30시간

## 문제

2025-2026 SRE 내러티브는 다음과 같이 정의되었습니다: "AI 에이전트가 인시던트를 분류하고, 인간이 해결 방안을 승인한다." AWS DevOps Agent, Resolve AI, NeuBird, Metoro, PagerDuty AIOps는 모두 이 아키텍처를 프로덕션 환경에 배포했습니다. 이 에이전트는 Prometheus 메트릭, Loki 로그, Tempo 트레이스, kube-state-metrics, 그리고 K8s 객체 지식 그래프를 읽습니다. 5분 이내에 텔레메트리 인용이 포함된 순위 기반 근본 원인 가설을 생성합니다. Slack을 통한 명시적인 인간 승인 없이는 절대 파괴적인 명령을 실행하지 않습니다.

대부분의 어려운 작업은 추론이 아닌 범위 설정과 안전성 확보에 있습니다. 에이전트는 기본적으로 읽기 전용 RBAC 표면, 강화된 MCP 도구 서버, 그리고 고려된 모든 명령과 실행된 명령에 대한 감사 로그가 필요합니다. 자신의 한계를 인지하고 에스컬레이션할 수 있어야 합니다. 또한 OOM-kill 폭주로 인해 $5,000의 에이전트 비용이 발생하지 않을 정도로 저렴하게 운영되어야 합니다.

## 개념

에이전트는 지식 그래프 상에서 동작합니다. 노드는 K8s 오브젝트(Pods, Deployments, Services, Nodes, HPAs, PVCs)와 텔레메트리 소스(Prometheus 시리즈, Loki 스트림, Tempo 트레이스)로 구성됩니다. 엣지는 소유권(Pod -> ReplicaSet -> Deployment), 스케줄링(Pod -> Node), 관측(Pod -> Prometheus 시리즈)을 인코딩합니다. 이 그래프는 kube-state-metrics 동기화를 통해 최신 상태를 유지하며, 모든 경고 발생 시 재샘플링됩니다.

경고가 발생하면 에이전트는 영향을 받은 오브젝트부터 근본 원인을 분석합니다. 엣지를 따라 이동하며 관련 텔레메트리 조각(최근 15분 분량)을 추출하고 가설을 작성합니다. 가설은 증거(지원하는 텔레메트리 인용 수, 최신성, 구체성)에 따라 순위가 매겨집니다. 상위 3개 가설은 그래프 경로 시각화와 함께 Slack으로 전송되며, 복구 작업을 위한 승인 버튼이 포함됩니다.

복구 작업은 게이트 제어됩니다. 기본 허용 작업은 읽기 전용입니다. 파괴적 작업(스케일링 다운, 롤백, Pod 삭제)은 Slack 승인이 필요하며, ArgoCD 롤백 훅은 에이전트가 보유하지 않는 인증 토큰을 요구합니다. 감사 로그는 에이전트가 *고려한* 모든 명령어를 기록하므로(실행된 명령어뿐만 아니라), 검토 과정에서 놓친 사항을 포착할 수 있습니다.

## 아키텍처

```
PagerDuty / Alertmanager 웹훅
           |
           v
     FastAPI 수신기
           |
           v
   LangGraph 근본 원인 분석 에이전트
           |
           +---- 읽기 전용 MCP 도구 ----+
           |                             |
           v                             v
   K8s 지식 그래프              원격 측정 데이터 조각
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   소유권 + 스케줄링          최근 15분, 범위 지정
           |
           v
   가설 순위 결정 (증거 가중치)
           |
           v
   Slack 요약 + 승인 버튼
           |
           v (승인 시)
   ArgoCD 롤백 훅 / PagerDuty 에스컬레이션
           |
           v
   감사 로그: 고려된 vs 실행된, 모든 명령어
```

## 스택

- 관측 가능성 소스: Prometheus, Loki, Tempo, kube-state-metrics  
- 지식 그래프: Neo4j(관리형) 또는 K8s 객체 + 텔레메트리 엣지의 kuzu(임베디드)  
- 에이전트: 도구별 허용 목록(allow-list)이 있는 LangGraph, 기본적으로 읽기 전용  
- 도구 전송: StreamableHTTP를 통한 FastMCP; 승인 게이트 뒤에 파괴적 도구를 위한 별도 서버  
- 모델: 근본 원인 추론을 위한 Claude Sonnet 4.7, 로그 요약을 위한 Gemini 2.5 Flash  
- 복구: ArgoCD 롤백 웹훅, PagerDuty 에스컬레이션, Slack 승인 카드  
- 감사: 추가 전용 구조화 로그(고려됨, 실행됨, 승인됨, 결과)  
- 배포: 자체 좁은 RBAC 역할을 가진 K8s 배포; 별도 네임스페이스

## 빌드 방법

1. **그래프 수집.** 30초마다 kube-state-metrics를 Neo4j/kuzu와 동기화합니다. 노드: Pod, Deployment, Node, Service, PVC, HPA. 엣지: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES. 텔레메트리 오버레이 엣지: OBSERVED_BY (Pod가 Prometheus 시리즈에 의해 관찰됨).

2. **경고 수신기.** PagerDuty 또는 Alertmanager 웹후크를 수신하는 FastAPI 엔드포인트. 영향을 받은 객체(들) 및 SLO 위반을 추출합니다.

3. **읽기 전용 도구 인터페이스.** kubectl, Prometheus 쿼리, Loki logql, Tempo traceql을 FastMCP로 래핑합니다. 모든 도구는 좁은 RBAC 동사("get", "list", "describe")를 가집니다. 기본 서버에는 "delete", "exec", "scale"이 없습니다.

4. **근본 원인 분석 에이전트.** 세 개의 노드(`sample`, `walk`, `hypothesize`)로 구성된 LangGraph. `sample`은 최근 15분 간의 텔레메트리 슬라이스를 가져오고, `walk`는 인접 객체에 대해 그래프를 쿼리하며, `hypothesize`는 텔레메트리 인용과 함께 순위가 매겨진 근본 원인 후보를 작성합니다.

5. **증거 점수화.** 각 가설의 점수 = 최신성(recency) * 특이성(specificity) * 그래프 경로 길이 역수 * 인용 횟수. 상위 3개 결과를 반환합니다.

6. **Slack 요약.** 가설, 그래프 경로 시각화(서버 측에서 렌더링된 서브그래프 이미지), 최대 하나의 복구 작업을 위한 승인 버튼을 포함한 첨부 파일을 게시합니다.

7. **복구 게이트.** 파괴적 도구(스케일 다운, 롤백, 삭제)는 승인 토큰 뒤에 있는 두 번째 MCP 서버에 있습니다. 에이전트는 Slack 카드가 인간에 의해 승인된 후에만 호출할 수 있습니다.

8. **감사 로그.** 추가 전용 JSONL: 모든 후보 명령에 대해 고려 여부, 실행 여부, 승인자를 기록합니다. 매일 S3로 전송합니다.

9. **합성 인시던트 세트.** 20가지 시나리오 구축: OOMKill 연쇄, DNS 플랩, HPA 스래싱, PVC 가득 참, 노이즈 있는 이웃, 결함 있는 사이드카, 잘못된 ConfigMap 롤아웃, 인증서 갱신, 이미지 풀 백오프 등. 근본 원인 정확도와 가설 생성 시간으로 에이전트 성능을 평가합니다.

## 사용 방법

```
webhook: alert.pagerduty.com -> checkout-api SLO 위반, 오류율 14%
[graph]   영향: Deployment checkout-api (3개 Pod, Node ip-10-2-3-4)
[walk]    인접 항목: ReplicaSet checkout-api-abc, Service checkout-api,
           최근 롤아웃 14분 전
[sample]  prometheus 오류율 14%, 상승 추세; loki /api/v2/pay에서 500s 발생
[hypo]    #1 불량 롤아웃: 최신 이미지 checkout-api:v2.41에서 /healthz 실패
          근거: deploy.yaml (리비전 42), prometheus 오류율, loki 500 스택
[slack]   [v2.40으로 롤백]  [에스컬레이션]  [무시]
          (승인 필요; 에이전트는 단독으로 롤백하지 않음)
```

## Ship It

`outputs/skill-devops-agent.md`가 산출물입니다. K8s 클러스터와 알림 소스를 기반으로 에이전트는 순위가 매겨진 근본 원인 가설과 Slack 승인 기반 복구 흐름을 생성합니다.

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 시나리오 스위트에서의 RCA 정확도 | 20개의 합성 인시던트 중 ≥80% 정확한 근본 원인 식별 |
| 20 | 안전성 | 감사 로그에 Slack 승인 없이 파괴적 조치 가드가 절대 발동되지 않음 |
| 20 | 가설 생성 시간 | 알림 발생부터 Slack 브리핑까지 p50 기준 5분 미만 |
| 20 | 설명 가능성 | 모든 가설에 그래프 경로와 텔레메트리 인용 포함 |
| 15 | 통합 완성도 | PagerDuty, Slack, ArgoCD, Prometheus 종단간 작동 확인 |
| **100** | | |

## 연습 문제

1. AWS의 DevOps Agent가 데모로 보여주는 동일한 세 가지 인시던트에 대해 에이전트를 실행해 보세요. 나란히 비교하여 게시하고, 에이전트가 다른 결과를 보이는 지점을 보고하세요.

2. "근접 사고(near-miss)" 감사 기능을 추가하세요. 승인 없이 실행됐다면 파괴적이었을 명령을 에이전트가 *고려했던* 모든 경우를 표시하세요. 1주일 동안의 근접 사고율을 측정하세요.

3. 가설 모델을 Claude Sonnet 4.7에서 자체 호스팅 Llama 3.3 70B로 교체하세요. 근본 원인 분석(RCA) 정확도 변화와 인시던트당 비용을 측정하세요.

4. 인과 관계 필터(causal filter)를 구축하세요: 상관된 원격 분석(telemetry) 급증과 실제 근본 원인을 구분하세요. 20개 시나리오 레이블로 소형 분류기를 학습시키세요.

5. 롤백 드라이런(rollback dry-run) 기능을 추가하세요: 스테이징 클러스터에서 ArgoCD 롤백을 수행하고 동일한 매니페스트를 사용하세요. Slack 승인 버튼 전에 라이브 클러스터에서 롤백 계획을 검증하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| K8s 지식 그래프 | "클러스터 그래프" | 노드 = K8s 객체 + 텔레메트리 시리즈; 엣지 = 소유권, 스케줄링, 관찰 |
| 읽기 전용 기본 설정 | "범위 RBAC" | 에이전트의 서비스 계정은 get/list/describe 동사만 가짐; 파괴적 동사는 승인 뒤에 있는 별도 서버에 존재 |
| 감사 로그 | "고려된 vs 실행된" | 모든 후보 명령의 추가 전용 기록, 실행 여부, 승인자 정보 |
| 가설 순위 | "증거 점수" | 최신성 × 구체성 × 그래프 경로 길이 역수 × 인용 횟수 |
| Slack 승인 카드 | "HITL 게이트" | 복구 버튼이 있는 대화형 Slack 메시지; 인간이 클릭할 때까지 에이전트 진행 불가 |
| 텔레메트리 인용 | "증거 포인터" | 주장을 지원하는 Prometheus 쿼리, Loki 선택자 또는 Tempo 추적 URL |
| MTTR | "해결 시간" | 경고 발생부터 SLO 복구까지의 총 소요 시간 |

## 추가 자료

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — 2026년 기준 공식 문서
- [Resolve AI K8s 문제 해결](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — 경쟁사 참조 문서
- [NeuBird 시맨틱 모니터링](https://www.neubird.ai) — 시맨틱 그래프 접근법
- [Metoro AI SRE](https://metoro.io) — SLO 중심 프로덕션 프레임워크
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — 클러스터 상태 소스
- [LangGraph](https://langchain-ai.github.io/langgraph/) — 참조 에이전트 오케스트레이터
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP 서버 프레임워크
- [ArgoCD 롤백](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — 게이트된 복구 대상