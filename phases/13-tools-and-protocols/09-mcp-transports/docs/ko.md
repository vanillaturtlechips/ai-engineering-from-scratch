# MCP 전송 방식 — stdio vs Streamable HTTP vs SSE 마이그레이션

> stdio는 로컬에서만 작동하며 다른 곳에서는 작동하지 않습니다. Streamable HTTP(2025-03-26)는 원격 표준입니다. 기존 HTTP+SSE 전송 방식은 2026년 중반에 제거될 예정입니다. 잘못된 전송 방식을 선택하면 마이그레이션 비용이 발생하며, 올바른 방식을 선택하면 세션 연속성과 DNS-리바인딩 보호 기능을 갖춘 원격 호스팅 가능한 MCP 서버를 확보할 수 있습니다.

**유형:** 학습
**언어:** Python (표준 라이브러리, Streamable HTTP 엔드포인트 스켈레톤)
**선수 지식:** 13단계 · 07, 08 (MCP 서버 및 클라이언트)
**소요 시간:** ~45분

## 학습 목표

- 배포 형태(로컬 vs 원격, 단일 프로세스 vs 플릿)에 따라 stdio와 Streamable HTTP 중 적절한 방식을 선택.
- Streamable HTTP 단일 엔드포인트 패턴 구현: 요청에는 POST, 세션 스트림에는 GET 사용.
- `Origin` 검증과 세션 ID 의미 체계 강제 적용으로 DNS 리바인딩 공격 방지.
- 2026년 중반 제거 예정일 전에 레거시 HTTP+SSE 서버를 Streamable HTTP로 마이그레이션.

## 문제

첫 번째 MCP 원격 전송(2024-11)은 HTTP+SSE였습니다: 두 개의 엔드포인트가 있었는데, 하나는 클라이언트의 POST 요청을 위한 것이고 다른 하나는 서버-클라이언트 스트림을 위한 Server-Sent-Events 채널이었습니다. 이 방식은 작동했지만 번거로웠습니다: 세션당 두 개의 엔드포인트, 일부 CDN 앞에서 캐시 문제 발생, 그리고 일부 WAF(웹 애플리케이션 방화벽)에서 공격적으로 종료하는 장기 SSE 연결에 대한 강한 의존성.

2025-03-26 사양은 이를 Streamable HTTP로 대체했습니다: 하나의 엔드포인트, 클라이언트 요청에는 POST, 세션 스트림 설정에는 GET을 사용하며, 둘 다 `Mcp-Session-Id` 헤더를 공유합니다. 이후 구축되거나 마이그레이션된 모든 서버는 Streamable HTTP를 사용합니다. 이전 SSE 모드는 단계적으로 폐지되고 있습니다 — Atlassian Rovo는 2026년 6월 30일에, Keboola는 2026년 4월 1일에 제거했으며, 대부분의 남은 엔터프라이즈 서버도 2026년 말까지 폐지할 예정입니다.

그리고 로컬 서버에는 여전히 stdio가 중요합니다. Claude Desktop, VS Code 및 모든 IDE 형태의 클라이언트는 stdio를 통해 서버를 생성합니다. 올바른 정신 모델: "이 머신"에는 stdio, "네트워크 상"에는 Streamable HTTP. 상호 교차 없음.

## 개념

### stdio

- 자식 프로세스 전송. 클라이언트가 서버를 생성하고 stdin/stdout을 통해 통신.
- 한 줄에 하나의 JSON 객체. 개행으로 구분.
- 세션 ID 없음; 프로세스 ID가 세션 역할.
- 인증 불필요 (자식 프로세스는 부모 프로세스의 신뢰 경계를 상속).
- 원격 서버에는 절대 사용하지 않음 — SSH 또는 socat 터널링이 필요한 경우 Streamable HTTP 사용.

### Streamable HTTP

단일 엔드포인트 `/mcp` (또는 임의의 경로). 세 가지 HTTP 메서드 지원:

- **POST /mcp.** 클라이언트가 JSON-RPC 메시지를 전송. 서버는 단일 JSON 응답 또는 하나 이상의 응답을 포함하는 SSE 스트림으로 응답 (일괄 응답 및 해당 요청과 관련된 알림에 유용).
- **GET /mcp.** 클라이언트가 장기 SSE 채널을 엽니다. 서버는 클라이언트-서버 요청(샘플링, 알림, 유도)에 사용.
- **DELETE /mcp.** 클라이언트가 세션을 명시적으로 종료.

세션은 서버가 첫 응답에 설정하는 `Mcp-Session-Id` 헤더와 클라이언트가 이후 모든 요청에 에코하는 값으로 식별. 세션 ID는 반드시 암호학적으로 무작위(128+ 비트)여야 함; 클라이언트가 선택한 ID는 안전상 거부됨.

### 단일 엔드포인트 vs 두 엔드포인트

이전 사양에서 두 엔드포인트 모드는 2026년에도 여전히 호출 가능 — 사양은 이를 "레거시 호환"으로 선언. 하지만 모든 새 서버는 단일 엔드포인트를 사용해야 함. 공식 SDK는 단일 엔드포인트를 생성; 레거시 모드는 마이그레이션되지 않은 원격 서버와 통신할 때만 사용.

### `Origin` 검증 및 DNS 리바인딩

브라우저는 (현재) MCP 클라이언트가 아니지만, 공격자는 브라우저를 속여 `localhost:1234/mcp`에 POST를 전송하도록 웹페이지를 제작할 수 있음 — 사용자의 로컬 MCP 서버가 수신 대기 중인 경우. 서버가 `Origin`을 검사하지 않으면 브라우저의 동일 출처 정책이 `Origin: http://evil.com`을 유효한 크로스 오리진으로 처리하므로 보호되지 않음.

2025-11-25 사양은 서버가 허용 목록에 없는 `Origin`의 요청을 거부하도록 요구. 허용 목록에는 일반적으로 MCP 클라이언트 호스트(`https://claude.ai`, `vscode-webview://*`)와 로컬 UI용 localhost 변형이 포함.

### 세션 ID 라이프사이클

1. 클라이언트가 `Mcp-Session-Id` 없이 첫 요청을 전송.
2. 서버가 무작위 ID를 할당하고 응답 헤더에 `Mcp-Session-Id`를 설정.
3. 클라이언트가 이후 모든 요청과 스트림용 `GET /mcp`에 해당 헤더를 에코.
4. 서버는 세션을 취소할 수 있음; 클라이언트는 이후 요청에서 404를 수신하고 재초기화해야 함.
5. 클라이언트는 정리 종료를 위해 세션을 명시적으로 DELETE할 수 있음.

### Keepalive 및 재연결

SSE 연결이 끊길 수 있음. 클라이언트는 동일한 `Mcp-Session-Id`로 재-GET하여 재연결. 서버는 연결 끊김 동안 발생한 이벤트를 (합리적인 창 내에서) 큐에 저장하고 클라이언트가 에코한 `last-event-id` 헤더를 통해 재생해야 함.

Phase 13 · 13은 작업(Tasks)을 다루며, 장기 실행 작업이 전체 세션 재연결에도 생존할 수 있도록 함.

### 역호환성 프로브

이전 및 새 서버 모두 지원하려는 클라이언트:

1. `/mcp`에 POST.
2. 응답이 `200 OK`와 JSON 또는 SSE인 경우 Streamable HTTP.
3. 응답이 `200 OK`와 `Content-Type: text/event-stream` 및 보조 엔드포인트를 가리키는 `Location` 헤더인 경우 레거시 HTTP+SSE; `Location`을 따름.

### Cloudflare, ngrok 및 호스팅

2026년 프로덕션 원격 MCP 서버는 Cloudflare Workers(MCP Agents SDK 포함), Vercel Functions 또는 컨테이너화된 Node/Python에서 실행. 핵심: 호스팅이 SSE GET을 위한 장기 HTTP 연결을 지원해야 함. Vercel 무료 티어는 10초로 제한되며 부적합. Cloudflare Workers는 무한 스트림을 지원.

### 게이트웨이 구성

여러 MCP 서버를 게이트웨이로 프런트엔드할 때(Phase 13 · 17), 게이트웨이는 세션 ID를 재작성하고 업스트림을 다중화하는 단일 Streamable HTTP 엔드포인트. 도구는 게이트웨이 계층에서 병합; 클라이언트는 단일 논리적 서버를 인식.

### 전송 실패 모드

- **stdio SIGPIPE.** 자식 프로세스 종료 시 SIGPIPE 발생; 서버는 정상적으로 종료해야 함. 클라이언트는 EOF를 감지하고 세션을 종료 표시.
- **HTTP 502 / 504.** Cloudflare, nginx 및 기타 프록시는 업스트림 실패 시 이를 방출. Streamable HTTP 클라이언트는 짧은 백오프 후 한 번 재시도.
- **SSE 연결 끊김.** TCP RST, 프록시 타임아웃 또는 클라이언트 네트워크 변경으로 스트림 종료. 클라이언트는 `Mcp-Session-Id` 및 선택적 `last-event-id`로 재연결하여 재개.
- **세션 취소.** 서버가 세션 ID를 무효화; 클라이언트는 다음 요청에서 404를 수신. 클라이언트는 재핸드셰이크 필요.
- **클록 스큐.** 클라이언트의 리소스-TTL 계산이 서버와 달라짐. 클라이언트는 서버 타임스탬프를 권위 있는 것으로 취급.

### Streamable HTTP 우회 시기

일부 기업은 자체 네트워크 내에서 gRPC 또는 메시지 큐 전송 뒤에 MCP 서버를 배포. 이는 비표준 — MCP 사양은 이를 공식적으로 정의하지 않음. 게이트웨이는 내부적으로 gRPC를 사용하면서 MCP 클라이언트에 Streamable HTTP 표면을 노출할 수 있음. 외부 표면은 사양 준수를 유지; 게이트웨이가 변환을 담당.

## 사용 방법

`code/main.py`는 `http.server`(표준 라이브러리)를 사용하는 최소한의 Streamable HTTP 엔드포인트를 구현합니다. 이 파일은 `/mcp`에 대한 POST, GET, DELETE 요청을 처리하며, 첫 응답 시 `Mcp-Session-Id`를 설정하고, `Origin`을 검증하며, 허용 목록에 없는 출처의 요청은 거부합니다. 핸들러는 Lesson 07 노트 서버의 디스패치 로직을 재사용합니다.

주요 확인 사항:

- POST 핸들러는 JSON-RPC 본문을 읽고 디스패치한 후 JSON 응답을 작성합니다(단일 응답 변형; SSE 변종은 구조적으로 유사함).
- `Origin` 검사는 기본값인 `http://evil.example` 프로브를 거부하지만 `http://localhost`는 허용합니다.
- 세션 ID는 128비트 랜덤 16진수 문자열이며, 서버는 세션별 상태를 메모리에 유지합니다.

## Ship It

이 레슨은 `outputs/skill-mcp-transport-migrator.md`를 생성합니다. HTTP+SSE(레거시) MCP 서버가 주어졌을 때, 이 스킬은 세션 ID 연속성, Origin 검사, 역호환 프로브 지원을 갖춘 Streamable HTTP로의 마이그레이션 계획을 생성합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. `curl`에서 `initialize`를 POST하고 `Mcp-Session-Id` 응답 헤더를 관찰하세요. 헤더를 에코하여 두 번째 요청을 POST하고 세션 연속성을 검증하세요.

2. SSE 스트림을 여는 GET 핸들러를 추가하세요. 5초마다 하나의 `notifications/progress` 이벤트를 전송하세요. 동일한 세션 ID로 재-GET하여 재연결하고 서버가 이를 수락하는지 확인하세요.

3. `last-event-id` 재생 로직을 구현하세요. 재연결 시 해당 ID 이후 생성된 이벤트를 재생하세요.

4. 와일드카드 패턴(`https://*.example.com`)을 지원하도록 `Origin` 검증을 확장하세요. `https://app.example.com`은 허용하지만 `https://evil.example.com.attacker.net`은 거부하는지 확인하세요.

5. 공식 레지스트리에서 레거시 HTTP+SSE 서버를 가져와 마이그레이션 계획을 작성하세요: 엔드포인트 처리, 세션 ID 생성, 헤더 의미론에서 어떤 변경이 필요한지 설명하세요.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| stdio transport | "로컬 자식 프로세스" | stdin/stdout을 통한 JSON-RPC, 개행 문자 구분 |
| Streamable HTTP | "원격 전송 방식" | 단일 엔드포인트 POST + GET + 선택적 SSE, 2025-03-26 사양 |
| HTTP+SSE | "레거시" | 2026년 중반 제거 예정인 두 엔드포인트 모델 |
| `Mcp-Session-Id` | "세션 헤더" | 서버가 할당한 무작위 ID로, 이후 모든 요청에 에코됨 |
| `Origin` 허용 목록 | "DNS 리바인딩 방어" | 승인되지 않은 Origin의 요청 거부 |
| Single endpoint | "하나의 URL" | `/mcp`가 모든 세션 작업에 대한 POST/GET/DELETE 처리 |
| `last-event-id` | "SSE 재생" | 이벤트 누락 없이 중단된 스트림 재개에 사용되는 헤더 |
| Backwards-compat probe | "구형 vs 신형 감지" | 전송 방식을 자동 선택하는 클라이언트 응답 형태 검사 |
| Long-lived HTTP | "SSE 스트리밍" | 단일 TCP 연결에서 서버가 분 또는 시간 단위로 이벤트 푸시 |
| Session revocation | "강제 재초기화" | 서버가 세션 ID를 무효화; 클라이언트는 다시 핸드셰이크 필요 |

## 추가 자료

- [MCP — 기본 전송 사양 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio 및 Streamable HTTP에 대한 표준 참조 문서
- [MCP — 기본 전송 사양 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — Streamable HTTP를 도입한 개정판
- [Cloudflare — MCP 전송](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers 호스팅 Streamable HTTP 패턴
- [AWS — MCP 전송 메커니즘](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — 배포 형태 간 비교
- [Atlassian — HTTP+SSE 사용 중단 공지](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 구체적인 마이그레이션 마감일 예시