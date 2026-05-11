# 프로덕션 환경에서의 MCP 인증 — DCR, JWKS 회전, iii 프리미티브 기반 대상(Audience) 고정 토큰

> 레슨 16에서는 OAuth 2.1 상태 머신을 메모리에 구축했습니다. 2026년까지 실제 조직에 배포하는 모든 MCP 서버는 프로덕션 인증 뒤에 위치합니다: 동적 클라이언트 등록(RFC 7591), 인증 서버 메타데이터 검색(RFC 8414), 새벽 3시에 토큰 유효성 검사를 중단하지 않는 JWKS 회전, 그리고 혼동된 대리인(Confused Deputy) 재사용을 거부하는 대상(Audience) 고정 토큰. 이 레슨에서는 iii 프리미티브를 통해 이 모든 것을 통합합니다 — HTTP 및 cron용 `iii.registerTrigger`, 인증 로직용 `iii.registerFunction`, 캐시된 키용 `state::set/get` — 따라서 인증 표면이 엔진 내 다른 모든 워크로드와 마찬가지로 관찰 가능, 재시작 가능, 재생 가능하도록 구현됩니다.

**유형:** 구축
**언어:** Python (표준 라이브러리, 레슨 환경을 위해 iii 프리미티브 모킹)
**선수 조건:** 13·16 단계 (OAuth 2.1 상태 머신), 13·17 단계 (게이트웨이)
**소요 시간:** ~90분

## 학습 목표

- RFC 8414 메타데이터를 통해 인증 서버를 발견하고 계약을 검증합니다.
- RFC 7591 동적 클라이언트 등록을 구현하여 MCP 클라이언트가 관리자 개입 없이 등록되도록 합니다.
- 크론 트리거를 사용하여 JWKS 키를 캐싱하고 순환시켜 서명 검증이 키 롤오버 후에도 유지되도록 합니다.
- RFC 8707 리소스 표시자를 사용하여 토큰을 단일 MCP 리소스에 고정시키고 혼동된 대리인(confused-deputy) 재사용을 거부합니다.
- 모든 엔드포인트와 백그라운드 작업을 iii 프리미티브로 연결합니다. HTTP 트리거, 크론 트리거, 명명된 함수, `state::*` 읽기를 사용하여 단일 재시작으로 인증 표면을 재구성할 수 있도록 합니다.
- IdP(Identity Provider) 기능 매트릭스를 읽고, IdP가 MCP의 인증 프로필을 충족할 수 없을 때 배포를 거부합니다.

## 문제

Lesson 16 시뮬레이터는 OAuth 2.1을 메모리 내에서 실행합니다. 프로덕션 환경에서는 메모리 전용 시뮬레이터가 포착하지 못하는 세 가지 운영상의 차이가 있습니다.

첫 번째 차이는 등록(enrollment)입니다. 실제 조직에서는 수백 대의 MCP 서버와 수천 대의 MCP 클라이언트를 운영합니다. 운영자는 모든 Cursor 사용자를 OAuth 클라이언트로 수동 등록하지 않습니다. RFC 7591 동적 클라이언트 등록을 통해 클라이언트는 인증 서버에 `POST /register`를 전송하고 즉시 `client_id`(및 선택적으로 `client_secret`)를 받을 수 있습니다. 서버는 RFC 8414 메타데이터에 `registration_endpoint`를 게시하며, 클라이언트는 대역 외 구성 없이도 이를 발견할 수 있습니다.

두 번째 차이는 키 순환(key rotation)입니다. JWT 검증은 인증 서버의 서명 키인 JSON Web Key Set(JWKS)에 의존합니다. 인증 서버는 이 키들을 주기적으로(예: 시간 단위, 사고 대응 시 더 빠르게) 순환합니다. 부팅 시 JWKS를 한 번만 가져오는 MCP 서버는 키 순환 주기까지 정상적으로 검증되지만, 이후에는 재시작할 때까지 모든 요청이 실패합니다. 프로덕션 환경에서는 JWKS를 캐시된 값으로 구성하고, 이전 키가 만료되기 전에 캐시를 갱신하는 작업을 연결합니다. 또한 캐시 미스 시 폴백(fallback) 가져오기를 구현하여 캐시보다 최신 키로 서명된 토큰이 도착하는 경우를 처리합니다.

세 번째 차이는 대상(audience) 바인딩입니다. Lesson 16에서는 RFC 8707 리소스 표시자를 소개했습니다. 프로덕션 환경에서는 이 표시자가 모든 요청에서 강제적인 클레임 검사로 작동합니다. MCP 서버는 `token.aud`를 자체 정규화된 리소스 URL과 비교하고, 불일치 시 HTTP 401로 거부합니다. 이는 동일한 신뢰 메시지에 있는 다른 서버에 대해 상위 MCP 서버(또는 특정 서버용 토큰을 보유한 악성 클라이언트)가 토큰을 재전송하는 것을 방지하는 유일한 방어 수단입니다.

이 레슨에서는 이러한 차이들을 모두 iii 프리미티브로 처리합니다. 메타데이터 문서는 HTTP 트리거로, 함수의 출력을 반환합니다. JWKS 순환은 `auth::rotate-jwks`를 호출하는 크론 트리거로, `state::set("auth/jwks/<issuer>", ...)`에 씁니다. JWT 검증은 다른 함수들이 `iii.trigger("auth::validate-jwt", token)`을 통해 호출하는 함수입니다. MCP 서버 자체도 검증 후 디스패치하는 또 다른 HTTP 트리거입니다. 엔진을 재시작하면 트리거 레지스트리가 재구성되고, 상태는 유지되며, 인증 표면은 수동 조정 없이도 운영됩니다.

## 개념

### RFC 8414 — OAuth 인증 서버 메타데이터

`/.well-known/oauth-authorization-server` 문서의 JSON은 클라이언트가 필요한 모든 정보를 설명합니다:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

MCP 리소스 URL을 받은 클라이언트는 RFC 9728의 `oauth-protected-resource`(리소스 서버 문서)에서 발급자(issuer)를 확인한 후, 이 RFC의 `oauth-authorization-server`에서 모든 엔드포인트를 확인합니다. 클라이언트는 인증 URL을 하드코딩하지 않습니다.

MCP 서버가 IdP를 신뢰하기 전에 확인하는 계약 조건:

- `code_challenge_methods_supported`에 `S256`(RFC 7636의 PKCE)이 포함되어야 합니다.
- `grant_types_supported`에 `authorization_code`가 포함되어야 하며, `password`와 `implicit`는 거부해야 합니다.
- `registration_endpoint`가 존재해야 합니다(RFC 7591 지원).
- `response_types_supported`는 OAuth 2.1에 따라 정확히 `["code"]`여야 합니다.

이 중 하나라도 누락되면 MCP 서버는 해당 IdP에 대한 배포를 거부합니다. 코드가 아닌 배포 매니페스트가 잘못된 것입니다.

### RFC 9728 (복습) — 보호된 리소스 메타데이터

레슨 16에서 다룬 RFC 9728입니다. 프로덕션 환경의 차이점: 이 문서는 클라이언트가 *이* MCP 서버가 신뢰하는 인증 서버를 찾는 유일한 위치입니다. 단일 MCP 서버는 여러 IdP(예: 직원용, 파트너용)의 토큰을 수락할 수 있습니다. RFC 9728은 해당 집합을 선언하고, RFC 8414는 각 IdP가 지원하는 기능을 문서화합니다.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — 동적 클라이언트 등록

DCR(Dynamic Client Registration)이 없으면 모든 MCP 클라이언트(Cursor, Claude Desktop, 커스텀 에이전트)는 IdP 관리자와 사전 협의가 필요합니다. DCR을 사용하면 클라이언트는 다음 JSON을 POST로 전송합니다:

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

서버는 `client_id`와 이후 업데이트를 위한 `registration_access_token`으로 응답합니다:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none`은 사용자 디바이스에서 실행되는 MCP 클라이언트에 적합한 기본값입니다. `client_secret` 없이 `client_id`만 발급되며, PKCE가 공개 클라이언트에 필요한 소유 증명을 제공합니다.

프로덕션 환경의 3가지 함정:

- 등록 엔드포인트는 소스 IP별로 요청 수를 제한해야 합니다. 그렇지 않으면 악의적인 행위자가 수백만 개의 가짜 등록을 생성하여 `client_id` 네임스페이스를 고갈시킬 수 있습니다. iii는 이를 쉽게 구현합니다: 등록 HTTP 트리거는 `auth::rate-limit` 함수를 호출한 후 등록 처리기로 전달합니다.
- `software_statement`(서명된 JWT로 클라이언트를 보증하는 문서)는 일부 엔터프라이즈 IdP에서 필수입니다. 이 레슨의 모의 구현에서는 생략되지만, 프로덕션에서는 localhost 리다이렉트 URI가 아닌 경우 서명되지 않은 등록을 거부하는 검증 단계를 추가합니다.
- `registration_access_token`은 평문이 아닌 해시로 저장해야 합니다. 이 토큰이 유출되면 공격자가 클라이언트의 리다이렉트 URI를 재작성할 수 있습니다.

### RFC 8707 (복습) — 리소스 지시자

레슨 16에서 다룬 내용입니다. 프로덕션 규칙: 모든 토큰 요청에는 `resource=<canonical-mcp-url>`이 포함되어야 하며, MCP 서버는 모든 요청에서 `token.aud`가 자신의 리소스 URL과 일치하는지 검증해야 합니다. MCP 서버가 `https://notes.example.com/mcp`로 접근 가능하다면, 표준 URL은 `https://notes.example.com`입니다. 경로 구성 요소는 제외되어 단일 서버가 여러 경로를 하나의 대상(audience)으로 호스팅할 수 있습니다.

### RFC 7636 (복습) — PKCE

PKCE는 OAuth 2.1에서 필수입니다. 이 레슨의 인증 코드 흐름은 항상 `code_challenge`와 `code_verifier`를 포함합니다. 서버는 검증자가 없거나 저장된 챌린지와 해시 값이 일치하지 않는 토큰 요청을 거부합니다.

### MCP 사양 2025-11-25 인증 프로파일

MCP 사양(2025-11-25)은 MCP 서버의 인증 계층이 수행해야 할 작업을 정확히 정의합니다:

- `/.well-known/oauth-protected-resource`(RFC 9728)를 게시해야 합니다.
- `Authorization: Bearer ...`를 통해서만 토큰을 수락해야 합니다.
- 요청별로 `aud`, `iss`, `exp`, 필수 스코프를 검증해야 합니다.
- 모든 401 및 403 응답에 `WWW-Authenticate`를 포함해야 하며, `Bearer error=...`와 함께 `scope=` 및 `resource=` 매개변수를 포함해야 합니다.
- `aud`가 표준 리소스와 일치하지 않는 토큰을 거부해야 합니다.
- `iss`가 보호된 리소스 메타데이터의 `authorization_servers` 목록에 없는 토큰을 거부해야 합니다.

OAuth 2.1 초안이 기반이며, RFC 8414/7591/8707/9728 + RFC 7636가 표면이고, MCP 사양이 프로파일입니다.

### IdP 기능 매트릭스

모든 IdP가 전체 MCP 프로파일을 지원하는 것은 아닙니다. 아래 매트릭스는 2025-11-25 사양 기준 실제 기능 상태를 문서화합니다. 이는 *배포 게이트*이며, 권장 사항이 아닙니다.

| IdP 범주 | RFC 8414 메타데이터 | RFC 7591 DCR | RFC 8707 리소스 | RFC 7636 S256 PKCE | 비고 |
|---|---|---|---|---|---|
| 자체 호스팅(키클록) | 예 | 예 | 예(24.x 이후) | 예 | 이 레슨의 MCP 프로파일 참조 IdP; 모든 RFC를 종단간 지원. |
| 엔터프라이즈 SSO(MS Entra ID) | 예 | 예(프리미엄 계층) | 예 | 예 | DCR 가용성은 테넌트 계층에 따라 다름; 배포 전 대상 테넌트에서 확인 필요. |
| 엔터프라이즈 SSO(Okta) | 예 | 예(Okta CIC / Auth0) | 예 | 예 | Auth0(현재 Okta CIC)에서 DCR 사용 가능; 클래식 Okta 조직은 관리자 사전 등록 필요. |
| 소셜 로그인 IdP(일반적) | 다양함 | 거의 없음 | 거의 없음 | 예 | 대부분의 소셜 IdP는 클라이언트를 정적 파트너로 취급; DCR에 의존하지 말 것. 신원 소스로만 사용하고, MCP 인식 인증 서버를 별도로 계층화. |
| 커스텀/자체 제작 | 의존적 | 의존적 | 의존적 | 의존적 | 자체 제작 시 전체 프로파일을 구현해야 함. 위 4개 RFC 중 하나라도 생략하면 MCP 인증 계약이 깨짐. |

배포 매니페스트 거부 규칙: 선택한 IdP가 `registration_endpoint`를 반환하지 않거나 `code_challenge_methods_supported`에 `S256`을 나열하지 않으면 MCP 서버는 시작을 거부합니다. 저하 모드(degraded mode)는 없습니다.

### iii를 사용한 JWKS 회전 패턴

프로덕션 실패 모드는 오래된 JWKS 캐시입니다. 크론 트리거와 `state::*` 캐시로 해결합니다:

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

6시간마다 크론 트리거는 `auth::rotate-jwks`를 호출하여 `<issuer>/.well-known/jwks.json`을 가져오고 `state::set("auth/jwks/<issuer>", {keys, fetched_at})`에 씁니다. 검증기는 `state::get`에서 읽습니다. 토큰의 `kid`가 캐시에 없으면 동기식 `auth::rotate-jwks` 호출이 폴백으로 트리거됩니다. 이는 두 가지 경우를 동시에 처리합니다: 예약된 회전(크론)과 키 오버랩 창(동기식 폴백).

상태 구조:

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

두 개의 키를 동시에 보유하는 것이 정상 상태입니다. 인증 서버는 이전 키(`k_2026_03`)를 폐기하기 전에 다음 키(`k_2026_04`)를 도입하여 토큰이 만료될 때까지 유효성을 유지합니다. 캐시는 합집합을 보유하며, 검증기는 `kid`로 선택합니다.

### iii 프리미티브 연결(이 레슨의 핵심)

5개의 프리미티브가 인증 표면을 구성합니다:

```python
# 1. RFC 8414 메타데이터 문서
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 동적 클라이언트 등록
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT 검증을 호출 가능한 함수로 (리소스 서버가 트리거)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. 점진적 스코프를 위한 스텝업 발급 (L16의 SEP-835)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. 크론 기반 JWKS 회전
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

MCP 서버 자체는 검증을 직접 호출하지 않습니다. 다음과 같이 수행합니다:

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

이 간접 호출은 iii의 핵심 전략입니다. 내일은 검증기를 두 IdP에 병렬로 조회하는 팬아웃으로 교체하거나, 스팬 에미터를 추가하거나, 긍정적 검증을 캐싱할 수 있습니다. MCP 서버는 변경되지 않습니다.

### 대상 바인딩을 통한 혼동된 대리인 공격 시나리오

서버 A(`notes.example.com`)와 서버 B(`tasks.example.com`)가 동일한 인증 서버에 등록되었습니다. 서버 A가 침해당했습니다. 공격자는 사용자의 notes 토큰을 탈취하여 서버 B에 재사용합니다.

서버 B의 검증기:

1. JWT 디코딩, `kid`로 JWKS 가져오기, 서명 검증.
2. `iss`를 보호된 리소스 메타데이터의 `authorization_servers`와 비교. (통과 — 동일한 IdP.)
3. `aud == "https://tasks.example.com"` 확인. (실패 — 토큰의 `aud`는 `https://notes.example.com`.)
4. `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch"`와 함께 401 반환.

대상 클레임은 프로토콜 계층에서 이 공격을 방어하는 유일한 수단입니다. 성능을 위해 생략하는 것이 가장 흔한 프로덕션 실수입니다. 검증기는 세션 시작 시뿐만 아니라 모든 요청에서 실행되어야 합니다.

### 실패 모드

- **오래된 JWKS.** 키 회전 후 검증기가 유효한 토큰을 거부합니다. 해결책은 위의 크론+폴백 패턴입니다. 갱신 작업 없이 JWKS를 캐시하지 마십시오.
- **누락된 `aud` 클레임.** 일부 IdP는 토큰 요청에 `resource`가 없으면 `aud`를 생략합니다. 검증기는 `aud`가 없는 토큰을 거부해야 하며, 부재를 와일드카드로 취급해서는 안 됩니다.
- **스코프 업그레이드 경쟁 조건.** 동일한 사용자에 대한 두 개의 동시 스텝업 흐름이 모두 성공하여 서로 다른 스코프를 가진 두 개의 액세스 토큰이 생성될 수 있습니다. 검증기는 요청에 제시된 토큰을 사용해야 하며, "사용자의 현재 스코프"를 조회해서는 안 됩니다. 이는 TOCTOU(경쟁 조건) 창을 생성합니다.
- **등록 토큰 도난.** 유출된 `registration_access_token`은 공격자가 리다이렉트 URI를 재작성할 수 있게 합니다. 이 토큰을 해시로 저장하고, 클라이언트가 모든 업데이트 시 평문을 제시하도록 요구하며, 의심 시 회전하십시오.
- **`iss` 고정되지 않음.** 모든 `iss`를 수락하는 검증기는 공격자가 자체 인증 서버를 구축하고 대상 대상에 대한 클라이언트를 등록한 후 토큰을 발급할 수 있게 합니다. 보호된 리소스 메타데이터의 `authorization_servers` 목록은 허용 목록입니다. 이를 강제하십시오.

## 사용 방법

`code/main.py`는 표준 라이브러리 Python과 `iii_mock` 레지스트리를 사용하여 전체 프로덕션 흐름을 실행합니다. `iii_mock`은 `iii.registerFunction`, `iii.registerTrigger`, `iii.trigger`, `state::set/get`을 모방합니다. 흐름은 다음과 같습니다:

1. 인증 서버가 `/.well-known/oauth-authorization-server`에 RFC 8414 메타데이터를 게시합니다.
2. MCP 클라이언트가 메타데이터 엔드포인트를 호출하고 등록 엔드포인트를 발견합니다.
3. MCP 클라이언트가 `/register`(RFC 7591)에 POST 요청을 보내고 `client_id`를 수신합니다.
4. MCP 클라이언트가 PKCE 보호 인증 코드 흐름(RFC 7636)을 실행하며 `resource` 표시자(RFC 8707)를 사용합니다.
5. MCP 클라이언트가 `Authorization: Bearer ...`로 MCP 서버의 도구를 호출합니다.
6. MCP 서버가 `auth::validate-jwt`를 트리거하고, 이는 `state::get`에서 JWKS를 읽습니다.
7. 크론 트리거가 `auth::rotate-jwks`를 실행하여 상태의 JWKS를 교체합니다.
8. 다음 호출은 재시작 없이 새 키로 검증됩니다.
9. 다른 MCP 리소스에 대한 혼란 대리인 시도(Confused-deputy attempt)는 대상(Audience) 불일치로 401을 반환합니다.

여기서 모의 JWT는 공유 비밀(shared secret)을 사용하는 HS256을 사용합니다(따라서 표준 라이브러리만으로 실행 가능). 프로덕션에서는 RS256 또는 EdDSA를 위의 JWKS 패턴과 함께 사용하며, 검증 로직은 그 외에는 동일합니다.

## Ship It

이 레슨은 `outputs/skill-mcp-auth-iii.md`를 생성합니다. MCP 서버 설정과 IdP 기능 집합이 주어졌을 때, 스킬은 iii 프리미티브를 방출하여 등록, JWKS 회전 일정, 스코프 매핑, 그리고 IdP가 전체 RFC 프로파일을 지원하지 않을 때 적용할 거부 규칙을 정의합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 9단계 흐름을 추적하세요. `auth::rotate-jwks`가 데이터를 덮어쓰기 직전에 `state::get`이 오래된 데이터를 반환하는 지점과, 다음 요청이 새 키로 유효성을 검사하는 방식을 확인하세요.

2. 보호 리소스 메타데이터의 `authorization_servers` 목록에 새로운 IdP(Identity Provider)를 추가하세요. 새 IdP로 서명한 토큰을 발급하고 검증기가 이를 수락하는지 확인하세요. 목록에 없는 IdP로 서명한 토큰을 발급하고 검증기가 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`로 거부하는지 확인하세요.

3. `auth::rate-limit`을 iii 함수로 구현하고 등록 HTTP 트리거 내에서 등록기(registrar)가 실행되기 전에 호출하세요. `state::set("auth/ratelimit/<ip>", ...)`에 저장된 소스 IP별 토큰 버킷(token-bucket)을 사용하세요.

4. RFC 7591을 읽고 본 강의의 `/register` 핸들러가 검증하지 않는 두 가지 필드를 식별하세요. 검증 로직을 추가하세요. (힌트: `software_statement` 및 `redirect_uris` URI 스킴.)

5. MCP 사양 2025-11-25의 인증 섹션을 읽으세요. 본 강의의 검증기가 현재 발행하지 않는 `WWW-Authenticate` 헤더에 대한 하나의 규범적 요구사항(normative requirement)을 찾아 추가하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| ASM | "OAuth 메타데이터 문서" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | "자체 서비스 클라이언트 등록" | RFC 7591 `POST /register` 흐름 |
| JWKS | "JWT 검증을 위한 공개 키" | JSON 웹 키 세트, `jwks_uri`에서 가져옴, `kid`로 인덱싱 |
| 리소스 지시자 | "대상(audience) 파라미터" | RFC 8707 `resource` 파라미터로 토큰을 특정 서버에 고정 |
| `aud` 클레임 | "대상" | 검증기가 표준 리소스 URL과 비교하는 JWT 클레임 |
| 혼동된 대리인 | "토큰 재생" | 서버 A를 위해 발급된 토큰이 서버 B에 제시되는 공격 |
| `iss` 허용 목록 | "신뢰할 수 있는 인증 서버" | 보호된 리소스 메타데이터의 `authorization_servers`에 명시된 집합 |
| 키 순환 | "JWKS 롤링" | 겹침 기간을 두고 서명 키를 주기적으로 교체 |
| 퍼블릭 클라이언트 | "네이티브 또는 브라우저 클라이언트" | `client_secret`이 없는 OAuth 클라이언트; PKCE로 보완 |
| `WWW-Authenticate` | "401/403 응답 헤더" | 클라이언트 복구를 유도하는 `Bearer error=...` 지시문 포함 |

## 추가 자료

- [MCP — 권한 부여 사양 (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — 이 레슨에서 구현하는 MCP 권한 부여 프로파일
- [RFC 8414 — OAuth 2.0 인증 서버 메타데이터](https://datatracker.ietf.org/doc/html/rfc8414) — 발견 계약
- [RFC 7591 — OAuth 2.0 동적 클라이언트 등록 프로토콜](https://datatracker.ietf.org/doc/html/rfc7591) — DCR
- [RFC 7636 — 코드 교환을 위한 증명 키(PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — 공개 클라이언트 소유 증명
- [RFC 8707 — OAuth 2.0 리소스 표시자](https://datatracker.ietf.org/doc/html/rfc8707) — 대상 고정
- [RFC 9728 — OAuth 2.0 보호 리소스 메타데이터](https://datatracker.ietf.org/doc/html/rfc9728) — 리소스 서버 발견
- [OAuth 2.1 초안](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 통합된 OAuth 기반 프로토콜