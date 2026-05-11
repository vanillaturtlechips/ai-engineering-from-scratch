# MCP 게이트웨이와 레지스트리 — 엔터프라이즈 제어 평면

> 엔터프라이즈는 모든 개발자가 임의의 MCP 서버를 설치하도록 방치할 수 없습니다. 게이트웨이는 인증, RBAC(역할 기반 접근 제어), 감사, 속도 제한, 캐싱, 도구 오염 감지를 중앙 집중화한 후, 병합된 도구 표면을 단일 MCP 엔드포인트로 노출합니다. 공식 MCP 레지스트리(Anthropic + GitHub + PulseMCP + Microsoft, 네임스페이스 검증됨)는 표준 업스트림입니다. 이 강의에서는 게이트웨이의 적절한 위치를 명명하고, 최소 구현을 안내하며, 2026년 벤더 환경을 조사합니다.

**유형:** 학습  
**언어:** Python (표준 라이브러리, 최소 게이트웨이)  
**선수 조건:** 13단계 · 15 (도구 오염), 13단계 · 16 (OAuth 2.1)  
**소요 시간:** ~45분

## 학습 목표

- MCP 게이트웨이가 위치하는 곳(MCP 클라이언트와 여러 백엔드 MCP 서버 사이) 설명.
- 5가지 게이트웨이 책임 구현: 인증(auth), 역할 기반 접근 제어(RBAC), 감사(audit), 속도 제한(rate limit), 정책(policy).
- 게이트웨이 계층에서 고정된 도구 해시(pinned-tool-hash) 매니페스트 적용.
- 공식 MCP 레지스트리(Official MCP Registry)와 메타레지스트리(Glama, MCPMarket, MCP.so, Smithery, LobeHub) 구분.

## 문제 정의

한 Fortune 500 기업은 30개의 승인된 MCP 서버, 5,000명의 개발자, 규정 준수 및 감사 요구사항, 그리고 중앙 집중식 정책을 원하는 보안 팀을 보유하고 있습니다. 모든 개발자가 IDE에 임의의 서버를 설치하는 것은 허용되지 않습니다.

게이트웨이 패턴:

1. 게이트웨이는 개발자들이 연결하는 단일 Streamable HTTP 엔드포인트로 실행됩니다.
2. 게이트웨이는 각 백엔드 MCP 서버에 대한 자격 증명을 보유합니다.
3. 모든 개발자 요청은 게이트웨이 자체의 OAuth를 통해 인증되고 범위가 지정됩니다.
4. 게이트웨이는 정책을 적용하면서 백엔드 서버로 호출을 라우팅합니다.
5. 감사를 위해 모든 호출이 로깅됩니다.

Cloudflare MCP 포털, Kong AI 게이트웨이, IBM ContextForge, MintMCP, TrueFoundry, Envoy AI 게이트웨이 — 2025-2026년에 출시된 모든 게이트웨이 또는 게이트웨이 기능.

한편, 공식 MCP 레지스트리는 게이트웨이가 가져올 수 있는 선별된, 네임스페이스 검증된, 역방향 DNS 이름 서버로 표준 업스트림으로 출시되었습니다. 메타레지스트리(Glama, MCPMarket, MCP.so, Smithery, LobeHub)는 여러 소스에 걸쳐 서버를 집계합니다.

## 개념

### 5가지 게이트웨이 책임

1. **인증(Auth).** 개발자를 식별하기 위한 OAuth 2.1; 사용자 역할에 매핑.
2. **RBAC(Role-Based Access Control).** 사용자별 정책: 어떤 서버, 어떤 도구, 어떤 스코프.
3. **감사(Audit).** 모든 호출은 누가, 무엇을, 언제, 결과 여부와 함께 기록.
4. **속도 제한(Rate limit).** 사용자/도구/서버별 제한으로 남용 방지.
5. **정책(Policy).** 오염된 설명 거부, Rule of Two 강제 적용, PII(개인 식별 정보) 삭제.

### 단일 엔드포인트로서의 게이트웨이

개발자에게 게이트웨이는 하나의 MCP 서버처럼 보입니다. 내부적으로는 N개의 백엔드로 라우팅합니다. 세션 ID(Phase 13 · 09)는 경계에서 재작성됩니다.

### 자격 증명 보관

개발자는 백엔드 토큰을 볼 수 없습니다. 게이트웨이가 이를 보유하거나(또는 이를 수행하는 ID 공급자에 프록시) 합니다. 게이트웨이에서 `notes:read` 권한을 가진 개발자는 게이트웨이의 백엔드 자격 증명으로 notes MCP 서버에 간접적으로 접근할 수 있지만, 이는 간접 접근을 바인딩하는 정책 하에서만 가능합니다.

### 게이트웨이에서의 도구 해시 고정

게이트웨이는 승인된 도구 설명(SHA256 해시)의 매니페스트를 보유합니다. 발견 시 각 백엔드의 `tools/list`를 가져와 매니페스트와 해시를 비교한 후, 설명이 변경된 도구는 제거합니다. 이는 Phase 13 · 15의 러그 풀 방어 메커니즘을 중앙에서 적용한 것입니다.

### 정책 코드화(Policy-as-code)

고급 게이트웨이는 OPA/Rego, Kyverno 또는 Styra로 정책을 표현합니다. "사용자 `alice`는 `acme` 조직의 리포지토리에서만 `github.open_pr`을 호출할 수 있음"과 같은 규칙은 선언적으로 인코딩됩니다. 단순 게이트웨이는 직접 코딩된 Python을 사용합니다. 두 형태 모두 유효합니다.

### 세션 인식 라우팅

사용자의 세션에 여러 서버가 혼합된 경우, 게이트웨이는 다중화합니다: 개발자의 단일 MCP 세션은 서버당 하나의 백엔드 세션을 보유합니다. 모든 백엔드의 알림은 게이트웨이를 통해 개발자의 세션으로 라우팅됩니다.

### 네임스페이스 병합

게이트웨이는 모든 백엔드의 도구 네임스페이스를 병합하며, 일반적으로 충돌 시 접두사를 추가합니다. `github.open_pr`, `notes.search`와 같이 라우팅이 명확해집니다.

### 레지스트리

- **공식 MCP 레지스트리(`registry.modelcontextprotocol.io`).** Anthropic, GitHub, PulseMCP, Microsoft의 관리 하에 출시. 네임스페이스 검증(역방향 DNS: `io.github.user/server`). 기본 품질 기준 사전 필터링.
- **Glama.** 많은 소스를 집계하는 검색 중심 메타 레지스트리.
- **MCPMarket.** 벤더 목록이 포함된 상업적 성향의 디렉터리.
- **MCP.so.** 커뮤니티 디렉터리; 공개 제출 가능.
- **Smithery.** 패키지 관리자 스타일의 설치 흐름.
- **LobeHub.** LobeChat 앱에 통합된 UI 레지스트리.

엔터프라이즈 게이트웨이는 기본적으로 공식 레지스트리에서 가져오며, 메타 레지스트리에서 관리자가 선별한 추가 항목을 허용하고, 고정되지 않은 모든 것을 거부합니다.

### 역방향 DNS 네이밍

공식 레지스트리는 공용 서버에 역방향 DNS 이름을 의무화합니다: `io.github.alice/notes`. 네임스페이스는 점유 방지 및 신뢰 위임 명확화를 제공합니다.

### 벤더 조사, 2026년 4월

| 벤더 | 강점 |
|------|------|
| Cloudflare MCP 포털 | 엣지 호스팅; OAuth 통합; 무료 티어 |
| Kong AI 게이트웨이 | K8s 네이티브; 세분화된 정책; OpenTelemetry 로깅 |
| IBM ContextForge | 엔터프라이즈 IAM; 규정 준수; 감사 내보내기 |
| TrueFoundry | DevOps 중심; 메트릭 우선 |
| MintMCP | 개발자 플랫폼 지향 |
| Envoy AI 게이트웨이 | 오픈 소스; 사용자 정의 필터 |

Phase 17(프로덕션 인프라)에서는 게이트웨이 운영에 대해 더 깊이 다룹니다.

## 사용 방법

`code/main.py`는 약 150줄로 구성된 최소 게이트웨이를 제공합니다: 가짜 Bearer 토큰으로 사용자를 인증하고, 사용자별 RBAC 정책을 유지하며, 두 개의 백엔드 MCP 서버로 요청을 라우팅하고, 모든 호출을 감사 로그에 기록하며, 속도 제한을 적용하고, 고정된 매니페스트와 설명 해시가 일치하지 않는 백엔드 툴은 거부합니다.

확인해야 할 사항:

- `RBAC` 딕셔너리는 `user_id`를 키로 사용하며 허용된 `server_tool` 항목을 포함합니다.
- `AUDIT_LOG`는 이벤트만 추가 가능한(append-only) 목록입니다.
- 속도 제한은 사용자별 토큰 버킷(token bucket)을 사용합니다.
- 고정된 매니페스트는 `server::tool -> hash` 형태의 딕셔너리입니다.

## Ship It

이 레슨은 `outputs/skill-gateway-bootstrap.md`를 생성합니다. 기업용 MCP 계획(사용자, 백엔드, 규정 준수)이 주어졌을 때, 스킬은 게이트웨이 구성 사양을 생성합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 허용된 사용자로 호출을 한 다음, 허용되지 않은 사용자로 호출을 하고, 마지막으로 요청 제한 초과 버스트를 수행하세요. 세 가지 흐름 모두를 검증하세요.

2. 클라이언트에 반환하기 전에 결과에서 PII(개인 식별 정보)를 삭제하는 정책을 추가하세요. SSN(사회보장번호) 형태의 문자열에 대해 간단한 정규식(regex)을 사용하고, 이메일 및 전화번호와 같은 누락된 부분을 기록하세요.

3. 감사 로그를 확장하여 OpenTelemetry GenAI 스팬을 생성하도록 수정하세요. 13단계 · 20에서 정확한 속성을 다룹니다.

4. 50명의 개발자로 구성된 팀을 위한 RBAC(Role-Based Access Control) 정책을 설계하세요. 백엔드는 5개(notes, github, postgres, jira, slack)입니다. 각 백엔드에서 읽기 전용 권한을 가진 사람은 누구이며, 쓰기 권한을 가진 사람은 누구인가요?

5. Cloudflare 엔터프라이즈 MCP 게시물을 처음부터 끝까지 읽으세요. Cloudflare가 제공하는 기능 중 이 stdlib 게이트웨이에서 지원하지 않는 기능 하나를 식별하세요.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| 게이트웨이(Gateway) | "MCP 프록시" | 클라이언트와 백엔드 사이의 중앙 집중식 서버 |
| 자격 증명 보관(Credential vaulting) | "백엔드 토큰은 서버 측에 유지" | 개발자는 업스트림 토큰을 절대 볼 수 없음 |
| 세션 인식 라우팅(Session-aware routing) | "멀티 백엔드 세션" | 게이트웨이는 개발자 세션당 N개의 백엔드 세션을 다중화 |
| 도구 해시 고정(Tool-hash pinning) | "승인된 매니페스트" | 승인된 모든 도구 설명의 SHA256 해시; 중앙에서 러그 풀 차단 |
| RBAC(Role-based access control) | "사용자별 정책" | 도구 및 서버에 대한 역할 기반 접근 제어 |
| 정책 코드화(Policy-as-code) | "선언적 규칙" | 게이트웨이에서 강제되는 OPA/Rego, Kyverno, Styra 정책 |
| 감사 로그(Audit log) | "누가, 무엇을, 언제" | 규정 준수를 위한 추가 전용 이벤트 로그 |
| 속도 제한(Rate limit) | "사용자별 토큰 버킷" | 남용 방지를 위한 분당 제한 |
| 공식 MCP 레지스트리(Official MCP Registry) | "표준 업스트림" | `registry.modelcontextprotocol.io`, 네임스페이스 검증 완료 |
| 역방향 DNS 네이밍(Reverse-DNS naming) | "레지스트리 네임스페이스" | `io.github.user/server` 규칙 |

## 추가 자료

- [공식 MCP 레지스트리](https://registry.modelcontextprotocol.io/) — 표준 업스트림, 네임스페이스 검증됨
- [Cloudflare — 엔터프라이즈 MCP](https://blog.cloudflare.com/enterprise-mcp/) — OAuth 및 정책 기반 게이트웨이 패턴
- [agentic-community — MCP 게이트웨이 레지스트리](https://github.com/agentic-community/mcp-gateway-registry) — 오픈소스 참조 게이트웨이
- [TrueFoundry — MCP 게이트웨이란?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — 기능 비교 기사
- [IBM — MCP 컨텍스트 포지](https://github.com/IBM/mcp-context-forge) — IBM의 엔터프라이즈 게이트웨이