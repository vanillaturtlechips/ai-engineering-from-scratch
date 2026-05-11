# 캡스톤 13 — 레지스트리 및 거버넌스를 갖춘 MCP 서버

> 모델 컨텍스트 프로토콜은 2026년에 더 이상 미래가 아닌 기본 도구 사용 사양으로 자리잡았습니다. Anthropic, OpenAI, Google 및 모든 주요 IDE에서 MCP 클라이언트를 제공합니다. Pinterest는 내부 MCP 서버 생태계를 공개했습니다. AAIF 레지스트리는 `.well-known`에서 기능 메타데이터를 공식화했습니다. AWS ECS는 참조 상태 비저장 배포를 발표했습니다. Block의 goose-agent는 동일한 프로토콜을 호스팅된 어시스턴트 내부에 통합했습니다. 2026년 프로덕션 아키텍처는 다음과 같습니다: 스트림 가능 HTTP 전송, OAuth 2.1 범위, OPA 정책 게이트, 플랫폼 팀이 서버를 검색, 검증, 활성화할 수 있는 레지스트리. 이를 엔드 투 엔드로 구축하세요.

**유형:** 캡스톤  
**언어:** Python (서버, FastMCP 경유) 또는 TypeScript (@modelcontextprotocol/sdk), Go (레지스트리 서비스)  
**선수 조건:** 11단계(LLM 엔지니어링), 13단계(도구 및 MCP), 14단계(에이전트), 17단계(인프라), 18단계(안전성)  
**연습 단계:** P11 · P13 · P14 · P17 · P18  
**소요 시간:** 25시간

## 문제

MCP는 도구 사용을 위한 공통 언어(lingua franca)가 되었습니다. Claude Code, Cursor 3, Amp, OpenCode, Gemini CLI 및 모든 관리형 에이전트가 이제 MCP 서버를 소비합니다. 프로덕션 환경의 과제는 서버 작성(FastMCP로 쉽게 해결 가능)이 아니라 엔터프라이즈 요구사항을 충족하면서 대규모로 배포하는 것입니다: 테넌트별 OAuth 범위, 파괴적 도구에 대한 OPA 정책, StreamableHTTP를 통한 무상태 확장성, 발견을 위한 레지스트리, 도구 호출별 감사 로그. Pinterest의 내부 MCP 생태계와 AAIF 레지스트리 사양은 2026년의 기준을 설정했습니다.

10개의 내부 도구(Postgres 읽기 전용, S3 목록 조회, Jira, Linear, Datadog 등)를 노출하는 MCP 서버, 플랫폼 발견을 위한 레지스트리 UI, 파괴적 도구에 대한 인간 승인 게이트를 구축할 것입니다. 부하 테스트는 StreamableHTTP의 수평 확장성을 입증하며, 감사 추적은 엔터프라이즈 보안 검토를 충족시킵니다.

## 개념

MCP 2026 개정판은 기본 전송 방식으로 StreamableHTTP를 규정합니다. 이전의 stdio-and-SSE 형태와 달리 StreamableHTTP는 기본적으로 무상태(stateless)입니다: 단일 HTTP 엔드포인트가 JSON-RPC 요청을 수신하고, 응답을 스트리밍하며, 알림을 위한 장기 연결을 지원합니다. 무상태 특성은 로드 밸런서 뒤에서 수평 확장이 가능함을 의미합니다.

인증은 OAuth 2.1을 사용하며, 도구별 스코프(per-tool scopes)가 적용됩니다. 토큰은 `jira:read`, `s3:list`, `postgres:query:readonly`와 같은 스코프를 포함합니다. MCP 서버는 세션 시작 시점뿐만 아니라 도구 호출 시점에도 스코프를 확인합니다. 고위험 도구의 경우, 서버가 마지막 N분 이내에 `approved:by:human`으로 승격되지 않은 호출은 거부합니다. 이 승격은 Slack 검토 카드에서 비롯됩니다.

레지스트리(registry)는 별도의 서비스입니다. 모든 MCP 서버는 `.well-known/mcp-capabilities` 문서를 통해 도구 매니페스트, 전송 URL, 인증 요구사항을 노출합니다. 레지스트리는 이를 폴링(polling), 검증, 인덱싱합니다. 플랫폼 팀은 레지스트리 UI를 사용하여 사용 가능한 도구, 필요한 스코프, 소유 팀을 확인할 수 있습니다.

## 아키텍처

```
MCP 클라이언트 (Claude Code, Cursor 3, ...)
          |
          v
HTTPS 기반 스트리밍 HTTP (JSON-RPC + 스트리밍)
          |
          v
로드 밸런서 뒤의 MCP 서버 (FastMCP)
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 목록 조회  Jira       Linear     Datadog
(읽기 전용) (페이징)     (읽기)     (읽기)     (쿼리)
          |
   +------+-------------+
   v                    v
OPA 정책 게이트웨이   파괴적 도구 MCP (별도 서버)
                        |
                        v
                   Slack을 통한 인간 승인
                        |
                        v
                   감사 로그 (추가 전용, 테넌트별)

  레지스트리 서비스
     |
     v  각 서버에서 GET /.well-known/mcp-capabilities
     v
     UI: 검색 / 검증 / 활성화-비활성화 / 소유권 관리
```

## 스택

- 서버 프레임워크: FastMCP (Python) 또는 `@modelcontextprotocol/sdk` (TypeScript)
- 전송 계층: HTTPS 기반 StreamableHTTP (무상태)
- 인증: SPIFFE / SPIRE를 통한 워크로드 ID 기반 OAuth 2.1
- 정책: 도구별 OPA / Rego 규칙; 요청별 정책 결정 서비스
- 레지스트리: 자체 호스팅, `.well-known/mcp-capabilities` 매니페스트 소비
- 인간 승인: 파괴적 도구에 대한 Slack 인터랙티브 메시지
- 배포: AWS ECS Fargate 또는 Fly.io, 테넌트별 단일 서버 또는 테넌트 스코핑 공유
- 감사: 테넌트별 버킷에 구조화된 JSONL, 호출별 계보 추적

## 빌드하기

1. **도구 표면.** 10개의 내부 도구를 노출: Postgres 읽기 전용 쿼리, S3 객체 목록 조회, Jira 검색/가져오기, Linear 검색/가져오기, Datadog 메트릭 쿼리, PagerDuty 온콜 조회, GitHub 읽기 전용, Notion 검색, Slack 검색, Salesforce 읽기. 각 도구는 타입화된 스키마와 범위 레이블을 가짐.

2. **FastMCP 서버.** 도구를 마운트. StreamableHTTP 전송 구성. OAuth 토큰 검증 및 범위 강제 적용을 위한 미들웨어 추가.

3. **OPA 정책.** 도구별 Rego 정책: 어떤 범위가 호출을 허용하는지, 어떤 PII 삭제 규칙이 적용되는지, 어떤 페이로드 크기 제한이 적용되는지. 모든 도구 호출 시 결정 서비스 호출.

4. **레지스트리 서비스.** 등록된 서버에서 `.well-known/mcp-capabilities`를 폴링하는 별도의 Go 또는 TS 서비스. JSON 스키마로 검증 후 목록/검색/검증/활성화-비활성화 UI 노출.

5. **기능 매니페스트.** 각 서버는 `.well-known/mcp-capabilities`를 노출: 도구 목록, 인증 요구사항, 전송 URL, 소유 팀, SLO.

6. **파괴적 도구 분리.** 상태를 변경하는 도구(Jira 생성, Linear 생성, Postgres 쓰기)는 두 번째 MCP 서버에서 실행. 더 엄격한 인증 흐름: 토큰은 15분 이내에 Slack 카드를 통해 `approved:by:human` 범위로 승격되어야 함.

7. **감사 로그.** 테넌트별 추가 전용 JSONL: `{timestamp, user, tool, args_redacted, response_redacted, outcome}`. 쓰기 전 Presidio를 통한 PII 삭제.

8. **부하 테스트.** StreamableHTTP에서 100개의 동시 클라이언트. 두 번째 복제본 추가로 수평 확장 시연; 세션 고정 없이 로드 밸런서가 재분배하는 모습 보여줌.

9. **준수 테스트.** 공식 MCP 준수 테스트 스위트를 두 서버 모두에 실행. 모든 필수 섹션 통과.

## 사용 방법

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## Ship It

`outputs/skill-mcp-server.md`는 결과물을 설명합니다. OAuth 2.1 스코프와 OPA 게이트가 적용된 내부 도구용 프로덕션급 MCP 서버 + 레지스트리 + 감사 레이어입니다.

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 사양 준수 | StreamableHTTP + 기능 매니페스트가 MCP 준수 테스트를 통과 |
| 20 | 보안 | 스코프 강제 적용, 모든 도구에 대한 OPA 적용 범위, 시크릿 관리 |
| 20 | 관측 가능성 | PII(개인정보) 마스킹이 적용된 도구별 호출 감사 로그 |
| 20 | 확장성 | 100개 클라이언트 부하 테스트 시 수평 확장 시연 |
| 15 | 레지스트리 UX | 발견 / 검증 / 활성화-비활성화 워크플로우 |
| **100** | | |

## 연습 문제

1. 새로운 도구(Confluence 검색)를 추가하세요. 코어 서버를 수정하지 않고 레지스트리 검증 흐름을 통해 배포하세요.

2. `email`, `ssn`, `phone`이라는 이름의 컬럼을 포함하는 Postgres 쿼리 결과를 마스킹하는 OPA 정책을 작성하세요. 프로브 쿼리로 실험하세요.

3. 로컬 대기 시간 측면에서 StreamableHTTP와 stdio를 벤치마킹하세요. 호출당 p50/p95 지표를 보고하세요.

4. 테넌트별 할당량(도구별 테넌트당 분당 최대 N회 호출)을 구현하세요. 두 번째 OPA 규칙을 통해 강제 적용하세요.

5. [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance)에서 MCP 준수 테스트 스위트를 실행하고 모든 실패를 수정하세요.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| StreamableHTTP | "2026 MCP 전송 프로토콜" | 상태 비저장 HTTP + 스트리밍; 네트워크 서버용 SSE + stdio 대체 |
| Capability manifest | "잘 알려진 문서" | `.well-known/mcp-capabilities`에 위치한 도구 목록, 인증, 전송 URL 포함 |
| OPA / Rego | "정책 엔진" | 외부 규칙에 대한 도구 호출 권한 부여를 위한 Open Policy Agent |
| Scope elevation | "인간 승인" | Slack 승인을 통해 부여되는 단기 유효 범위; 파괴적 도구 사용 시 필수 |
| Registry | "도구 검색" | Capability manifest에서 MCP 서버 인덱스를 제공하는 서비스 |
| Workload identity | "SPIFFE / SPIRE" | OAuth 토큰 발급을 위한 암호화 기반 서비스 신원 |
| Conformance suite | "스펙 테스트" | StreamableHTTP + 도구 매니페스트 정확성 검증을 위한 공식 MCP 테스트 세트 |

## 추가 자료

- [Model Context Protocol 2026 로드맵](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP, 기능 메타데이터, 레지스트리
- [AAIF MCP 레지스트리 사양](https://github.com/modelcontextprotocol/registry) — 2026 레지스트리 사양
- [AWS ECS 참조 배포](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 참조 프로덕션 배포
- [Pinterest 내부 MCP 생태계](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 참조 내부 배포
- [Block `goose` MCP 사용법](https://block.github.io/goose/) — 참조 에이전트 소비 패턴
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 서버 프레임워크
- [Open Policy Agent](https://www.openpolicyagent.org/) — 정책 엔진 참조
- [SPIFFE / SPIRE](https://spiffe.io) — 워크로드 신원 참조