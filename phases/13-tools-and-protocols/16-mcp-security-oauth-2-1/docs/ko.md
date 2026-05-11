# MCP 보안 II — OAuth 2.1, 리소스 지시자, 증분 스코프

> 원격 MCP 서버는 인증뿐만 아니라 권한 부여가 필요합니다. 2025-11-25 사양은 OAuth 2.1 + PKCE + 리소스 지시자(RFC 8707) + 보호된 리소스 메타데이터(RFC 9728)와 정렬됩니다. SEP-835는 403 WWW-Authenticate에서 단계적 권한 부여와 함께 증분 스코프 동의를 추가합니다. 이 레슨에서는 상태 머신으로서 단계적 흐름을 구현하여 모든 홉(hop)을 확인할 수 있습니다.

**유형:** 빌드
**언어:** Python (표준 라이브러리, OAuth 상태 머신 시뮬레이터)
**선수 조건:** 13단계 · 09 (전송), 13단계 · 15 (보안 I)
**소요 시간:** ~75분

## 학습 목표

- 리소스 서버와 인증 서버의 책임을 구분합니다.
- PKCE로 보호된 OAuth 2.1 인증 코드 흐름을 설명합니다.
- `resource`(RFC 8707)와 보호된 리소스 메타데이터(RFC 9728)를 사용하여 혼동된 대리인 공격(confused-deputy attacks)을 방지합니다.
- 단계적 권한 상승(step-up authorization)을 구현합니다: 서버가 403 응답과 함께 WWW-Authenticate를 통해 더 높은 스코프를 요청하고, 클라이언트가 사용자 동의를 다시 요청한 후 재시도합니다.

## 문제 정의

2025년 이전 MCP는 임시 API 키 또는 인증 없이 원격 서버를 출시했습니다. 2025-11-25 사양은 OAuth 2.1 프로필로 이 간극을 해결합니다.

세 가지 실제 요구사항:

- **일반 원격 서버.** 사용자가 Notion/GitHub/Gmail에 접근하는 원격 MCP 서버를 설치합니다. PKCE를 포함한 OAuth 2.1이 적합한 형태입니다.
- **범위 확장.** `notes:read` 권한을 부여받은 노트 서버가 특정 작업을 위해 `notes:write` 권한이 필요할 수 있습니다. 전체 인증 흐름을 다시 수행하지 않고, 단계적 권한 상승(SEP-835)으로 추가 범위를 요청합니다.
- **혼란스러운 대리인 방지.** 클라이언트가 서버 A용으로 발급된 토큰을 보유합니다. 서버 A가 악의적으로 이 토큰을 서버 B에 제시하려 할 때, 리소스 표시자(RFC 8707)가 토큰을 의도된 대상에만 고정합니다.

OAuth 2.1 자체는 새로운 것이 아닙니다. 새로운 것은 MCP의 프로필입니다: 특정 필수 흐름(인증 코드 + PKCE만 허용; 기본적으로 암시적/클라이언트 자격 증명 불가), 모든 토큰 요청에 리소스 표시자 필수 적용, 보호된 리소스 메타데이터 공개로 클라이언트가 접근 위치를 알 수 있도록 하는 것 등이 포함됩니다.

## 개념

### 역할

- **클라이언트.** MCP 클라이언트 (Claude Desktop, Cursor 등).
- **리소스 서버.** MCP 서버 (노트, GitHub, Postgres 등).
- **인증 서버.** 토큰을 발급. 리소스 서버와 동일한 서비스이거나 별도의 IdP(Auth0, Keycloak, Cognito)일 수 있음.

MCP 프로필에서 리소스 서버와 인증 서버는 동일한 호스트일 수 있지만 URL로 구분되어야 함.

### 인증 코드 + PKCE

흐름:

1. 클라이언트가 `code_verifier`(무작위)와 `code_challenge`(SHA256)를 생성.
2. 클라이언트가 사용자를 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`로 리디렉션.
3. 사용자가 동의. 인증 서버가 `redirect_uri?code=...`로 리디렉션.
4. 클라이언트가 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`로 POST.
5. 인증 서버가 저장된 챌린지와 `verifier` 해시를 검증한 후 액세스 토큰 발급.
6. 클라이언트가 리소스 서버에 요청할 때마다 `Authorization: Bearer ...` 헤더로 토큰 사용.

PKCE는 인증 코드 가로채기 공격을 방지. 리소스 표시자는 토큰이 다른 곳에서 유효하지 않도록 함.

### 보호된 리소스 메타데이터 (RFC 9728)

리소스 서버는 `.well-known/oauth-protected-resource` 문서를 게시:

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

클라이언트는 리소스 서버에서 인증 서버를 발견. 구성 감소 — 클라이언트는 리소스 URL만 필요.

### 리소스 표시자 (RFC 8707)

토큰 요청의 `resource` 파라미터는 토큰의 대상 청중을 고정. 발급된 토큰에는 `aud: "https://notes.example.com"` 포함. 이 토큰을 수신한 다른 MCP 서버는 `aud`를 확인하고 거부.

### 스코프 모델

스코프는 공백으로 구분된 문자열. 일반적인 MCP 규칙:

- `notes:read`, `notes:write`, `notes:delete`
- 관리자 기능용 `admin:*` (최소한으로 사용)
- 신원 확인을 위한 `profile:read`

스코프 선택은 최소 권한 원칙: 현재 필요한 것만 요청, 더 필요할 때 확장.

### 단계적 권한 상승 (SEP-835)

사용자가 `notes:read`를 승인. 이후 에이전트에게 노트 삭제를 요청. 서버가 응답:

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

클라이언트가 `insufficient_scope` 오류를 확인하고, 추가 스코프에 대한 동의 대화상자를 사용자에게 표시, 미니 OAuth 흐름 수행, 새 토큰으로 요청 재시도.

### 토큰 청중 검증

모든 요청 시: 서버는 `token.aud == self.resource_url` 확인. 불일치 시 401. 이는 서버 간 토큰 재사용을 방지.

### 단기 토큰 및 갱신

액세스 토큰은 단기(1시간 기본)로 설정되어야 함. 리프레시 토큰은 갱신 시마다 회전. 클라이언트는 백그라운드에서 자동 갱신을 처리.

### 토큰 전달 금지

샘플링 서버(Phase 13 · 11)는 클라이언트의 토큰을 다른 서비스에 전달해서는 안 됨. 샘플링 요청이 경계.

### 혼동된 대리인 방지

토큰은 `aud`에 바인딩. 클라이언트는 `client_id`에 바인딩. 모든 요청은 둘 다에 대해 검증. 사양은 MCP 이전 원격 도구 생태계에서 흔했던 "토큰 전달" 패턴을 명시적으로 금지.

### 클라이언트 ID 발견

각 MCP 클라이언트는 고정 URL에 메타데이터를 게시. 인증 서버는 클라이언트 메타데이터 문서를 가져와 리디렉션 URI와 연락처 정보를 발견. 이는 수동 클라이언트 등록을 제거.

### 게이트웨이와 OAuth

Phase 13 · 17은 기업 게이트웨이가 OAuth를 처리하는 방법을 보여줌: 게이트웨이는 업스트림 서버의 자격 증명을 보유, 클라이언트에 발급하는 토큰은 게이트웨이에서 생성, 업스트림 토큰은 게이트웨이를 벗어나지 않음. 이는 신뢰 모델을 전환 — 사용자는 게이트웨이에 한 번 인증, 게이트웨이는 N개 서버 인증을 처리.

## 사용 방법

`code/main.py`는 상태 머신으로서 전체 OAuth 2.1 단계적 인증(Step-up) 흐름을 시뮬레이션합니다. 다음 기능을 구현합니다:

- PKCE 코드 검증자(verifier)/도전(challenge) 생성.
- 리소스 지시자(resource indicator)를 포함한 인증 코드 흐름.
- 보호된 리소스 메타데이터 엔드포인트.
- 대상(audience) 확인을 통한 토큰 검증.
- `insufficient_scope`(불충분한 범위) 시 단계적 인증(Step-up).

이 레슨에는 HTTP 서버가 포함되지 않습니다. 상태 머신은 메모리 내에서 실행되므로 모든 단계를 추적할 수 있습니다. 13단계 · 17단계의 게이트웨이 레슨에서는 실제 전송 계층과 연동하는 방법을 다룹니다.

## Ship It

이 레슨은 `outputs/skill-oauth-scope-planner.md`를 생성합니다. 도구가 있는 원격 MCP 서버가 주어졌을 때, 스킬은 스코프 집합, 고정 규칙, 단계적 승격 정책을 설계합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 두 가지 스코프 단계 상승 흐름을 추적하세요. 단계 상승 시 어떤 홉(hop)이 반복되는지 확인하세요.

2. 리프레시 토큰 순환(refresh-token rotation)을 추가하세요: 모든 리프레시 시 새 리프레시 토큰을 발급하고 이전 토큰을 무효화하세요. 토큰이 순환된 후 도난당한 리프레시 토큰이 사용되는 시나리오를 시뮬레이션하고 실패하는지 확인하세요.

3. 보호된 리소스 메타데이터 엔드포인트를 stdlib `http.server`를 사용한 실제 HTTP 응답으로 구현하세요. 레슨 09의 `/mcp` 엔드포인트를 참고하세요.

4. GitHub MCP 서버를 위한 스코프 계층 구조를 설계하세요: `read repo`, `write PR`, `approve PR`, `merge PR`, `admin`. 각 레벨 간 단계 상승(step-up)을 사용하세요.

5. RFC 8707과 RFC 9728을 읽으세요. MCP가 RFC 예제와 다르게 사용하는 9728의 한 필드를 식별하세요. (힌트: `scopes_supported`와 관련됨)

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| OAuth 2.1 | "모던 OAuth" | PKCE를 의무화하고 암시적 흐름을 금지하는 통합 RFC |
| PKCE | "소유 증명" | 인증 코드 가로채기를 방지하는 코드 검증자 + 챌린지 |
| 리소스 지시자 | "토큰 대상" | RFC 8707 `resource` 매개변수로 토큰을 단일 서버에 고정 |
| 보호 리소스 메타데이터 | "디스커버리 문서" | RFC 9728 `.well-known/oauth-protected-resource` |
| 단계적 인증 | "점진적 동의" | SEP-835 흐름으로 필요 시 범위 추가 |
| `insufficient_scope` | "WWW-Authenticate가 포함된 403" | 더 큰 범위에 대한 재동의를 요청하는 서버 신호 |
| 혼동된 대리인 | "서비스 간 토큰 재사용" | 신뢰할 수 있는 보유자가 부적절하게 토큰을 전달하는 공격 |
| 단기 토큰 | "액세스 토큰 TTL" | 빠르게 만료되는 베어러 토큰; 리프레시 토큰으로 갱신 |
| 범위 계층 | "최소 권한 스택" | 단계별 승격이 가능한 점진적 범위 집합 |
| 클라이언트 ID 메타데이터 | "클라이언트 디스커버리 문서" | 클라이언트가 자체 OAuth 메타데이터를 게시하는 URL

## 추가 자료

- [MCP — 권한 부여 사양](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 표준 MCP OAuth 프로필
- [den.dev — MCP 11월 권한 부여 사양](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 변경 사항 설명
- [RFC 8707 — OAuth 2.0 리소스 지시자](https://datatracker.ietf.org/doc/html/rfc8707) — 대상 고정 RFC
- [RFC 9728 — OAuth 2.0 보호 리소스 메타데이터](https://datatracker.ietf.org/doc/html/rfc9728) — 발견 문서 RFC
- [Aembit — MCP OAuth 2.1, PKCE 및 AI 권한 부여의 미래](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 실용적인 단계적 흐름 설명