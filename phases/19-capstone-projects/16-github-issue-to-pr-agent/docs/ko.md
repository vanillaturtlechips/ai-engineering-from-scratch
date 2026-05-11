# 캡스톤 16 — GitHub 이슈-투-PR 자율 에이전트

> AWS Remote SWE 에이전트, Cursor 백그라운드 에이전트, OpenAI Codex 클라우드, Google Jules는 모두 동일한 2026년 제품 형태를 제공합니다: 이슈에 레이블을 지정하면 PR을 생성합니다. 클라우드 샌드박스에서 에이전트를 실행하고 테스트가 통과하는지 확인한 후 근거(rational)가 포함된 검토 준비 완료 PR을 게시합니다. 어려운 부분은 리포지토리의 빌드 환경을 자동으로 재현하는 것, 자격 증명 유출 방지, 리포지토리별 예산 강제 적용, 에이전트가 강제 푸시(force-push)를 할 수 없도록 하는 것입니다. 이 캡스톤은 자체 호스팅 버전을 구축하고 호스팅 대안과 비용 및 통과율을 비교합니다.

**유형:** 캡스톤  
**언어:** Python (에이전트), TypeScript (GitHub 앱), YAML (액션)  
**선수 조건:** 11단계(LLM 엔지니어링), 13단계(툴), 14단계(에이전트), 15단계(자율성), 17단계(인프라)  
**적용 단계:** P11 · P13 · P14 · P15 · P17  
**소요 시간:** 30시간

## 문제

비동기 클라우드 코딩 에이전트는 대화형 코딩 에이전트(캡스톤 01)와 별개의 제품 범주입니다. UX는 GitHub 라벨입니다. 이슈에 `@agent fix this` 라벨을 지정하면 클라우드 샌드박스에서 작업자가 실행되어 저장소를 복제하고, 테스트를 실행하며, 파일을 편집하고, 검증한 후 에이전트의 근거를 본문에 포함한 PR을 엽니다. 대화형 루프나 터미널은 없습니다. AWS Remote SWE 에이전트, 커서 백그라운드 에이전트, OpenAI Codex 클라우드, 구글 줄스(Google Jules), 팩토리 드로이드(Factory Droids)가 모두 이 방향으로 수렴하고 있습니다.

엔지니어링 과제는 구체적입니다: 환경 재현(에이전트는 캐시된 개발 이미지 없이 저장소를 처음부터 빌드해야 함), 불안정한 테스트(재실행 또는 격리 필요), 자격 증명 범위 지정(최소한의 세분화된 권한을 가진 GitHub 앱), 일일 저장소별 예산 강제, 강제 푸시 금지 정책입니다. 캡스톤은 통과율, 비용, 안전성을 호스팅된 대안과 비교하여 측정합니다.

## 개념

트리거는 GitHub 웹후크(issue 라벨 또는 PR 댓글)입니다. 디스패처는 ECS Fargate 또는 Lambda에 작업을 큐에 넣습니다. 워커는 리포지토리에서 추론된 일반적인 Dockerfile(언어, 프레임워크)을 사용하여 Daytona 또는 E2B 샌드박스에 리포지토리를 클론합니다. 에이전트는 Claude Opus 4.7 또는 GPT-5.4-Codex에 대해 미니 스위 에이전트(mini-swe-agent) 또는 SWE-agent v2 루프를 실행합니다. 이 과정은 반복됩니다: 코드 읽기, 수정 제안, 패치 적용, 테스트 실행.

검증은 게이트 단계입니다. PR이 열리기 전에 샌드박스에서 전체 CI가 통과해야 합니다. 커버리지 델타가 계산되며, 임계값을 초과하여 음수인 경우 PR이 열리지만 `needs-review` 라벨이 붙습니다. 에이전트는 PR 설명과 리뷰어가 후속 조치를 위해 핑할 수 있는 `@agent` 스레드로 근거를 게시합니다.

안전성은 두 가지 GitHub 표면을 통해 범위가 지정됩니다: 앱은 `workflows: read`와 좁은 리포지토리 내용/PR 범위를 가진 단기 설치 토큰을 제공합니다. 앱 권한이 아닌 브랜치 보호 규칙이 "main에 직접 쓰기 금지" 및 "강제 푸시 금지"를 강제합니다. 앱은 바이패스 목록에 절대 추가되지 않습니다. `.github/workflows`에 대한 경로 기반 읽기 전용 액세스는 실제 GitHub 앱 기본 요소가 아니므로, 워커에서 파일 수정에 대한 에이전트의 허용 목록(allow-list)이 이를 강제해야 합니다. 리포지토리당 일일 예산 상한은 디스패처에서 강제됩니다(예: 리포지토리당 하루 최대 5개 PR, PR당 $20).

## 아키텍처

```
GitHub 이슈 라벨 `@agent fix` 또는 PR 댓글
            |
            v
    GitHub 앱 웹훅 -> AWS Lambda 디스패처
            |
            v
    ECS Fargate 작업(또는 GitHub Actions 자체 호스팅 러너)
       - 리포지토리 풀
       - Dockerfile 추론(언어, 패키지 관리자)
       - 데이토나/Daytona 또는 E2B 샌드박스 + 대상 런타임
       - 클론 -> git worktree -> 에이전트 브랜치
            |
            v
    mini-swe-agent / SWE-agent v2 루프
       Claude Opus 4.7 또는 GPT-5.4-Codex
       도구: ripgrep, tree-sitter, 읽기/편집, 테스트 실행, git
            |
            v
    샌드박스 내 CI 통과 확인 + 커버리지 변화량 검사
            |
            v (검증 완료)
    git 푸시 + GitHub 앱을 통한 PR 생성
       PR 본문 = 근거 + diff 요약 + 추적 URL
       라벨: 검토 필요(needs-review)
            |
            v
    운영자 검토; 후속 조치를 위해 에이전트 @-멘션 가능
```

## 스택

- 트리거: 세분화된 토큰을 가진 GitHub 앱; Lambda 또는 Fly.io를 통한 웹훅 수신기
- 워커: ECS Fargate 작업(또는 GitHub Actions 자체 호스팅 러너)
- 샌드박스: 작업별 Daytona 개발 컨테이너 또는 E2B 샌드박스
- 에이전트 루프: mini-swe-agent 기준선 또는 Claude Opus 4.7/GPT-5.4-Codex 기반 SWE-agent v2
- 검색: tree-sitter 리포지토리 맵 + ripgrep
- 검증: 샌드박스 내 전체 CI + 커버리지 델타 게이트
- 관측 가능성: PR 본문에서 링크된 PR별 추적 아카이브를 갖춘 Langfuse
- 예산: 리포지토리별 일일 달러 상한선; 리포지토리당 하루 최대 PR 수

## 빌드 방법

1. **GitHub 앱.** 세분화된 설치 토큰 권한: issues 읽기+쓰기, pull_requests 쓰기, contents 읽기+쓰기, workflows 읽기. 브랜치 보호(이를 수행할 수 있는 유일한 표면)는 "main 브랜치에 직접 푸시 금지" 및 "강제 푸시 금지"를 강제합니다. 앱은 우회 목록에 포함되지 않습니다. 워커는 GitHub 앱 권한이 경로 범위가 아니므로 제안된 diff에 대한 허용 목록 검사로 ".github/workflows" 하위에 "쓰기 금지"를 강제합니다.

2. **웹훅 수신기.** Lambda 함수는 이슈 레이블/풀 리퀘스트 코멘트 웹훅을 수신합니다. 레이블 `@agent fix this`로 필터링합니다. SQS에 작업을 큐에 넣습니다.

3. **디스패처.** SQS에서 작업을 가져옵니다. 리포지토리별 일일 예산을 강제합니다. 리포지토리 URL, 이슈 내용, 신선한 Daytona 샌드박스를 사용하여 ECS Fargate 작업을 시작합니다.

4. **환경 추론.** 언어(Python, Node, Go, Rust)와 패키지 관리자(uv, pnpm, go mod, cargo)를 감지합니다. Dockerfile이 없으면 동적으로 생성합니다.

5. **에이전트 루프.** mini-swe-agent 또는 SWE-agent v2와 Claude Opus 4.7을 사용합니다. 도구: ripgrep, tree-sitter repo-map, read_file, edit_file, run_tests, git. 하드 제한: $20 비용, 30분 벽시계 시간, 30 에이전트 턴.

6. **검증.** 루프 종료 후 샌드박스 내에서 전체 테스트 스위트를 실행합니다. jacoco/coverage.py를 통해 커버리지 델타를 계산합니다. CI 실패 시: 중지, PR을 열지 않습니다. 커버리지가 2% 이상 감소 시: `needs-review` 레이블과 함께 PR을 엽니다.

7. **PR 게시.** 에이전트 브랜치를 푸시합니다. GitHub API를 통해 다음 내용을 포함한 PR을 엽니다: 제목, 근거, diff 요약, 추적 URL, 비용, 턴 수.

8. **자격 증명 위생.** 워커는 단기 GitHub 앱 설치 토큰으로 실행됩니다. 로그는 보관 전 비밀 정보를 제거합니다.

9. **평가.** 다양한 난이도의 30개 시드 내부 이슈. 통과율, PR 품질(diff 크기, 스타일, 커버리지), 비용, 지연 시간을 측정합니다. 동일한 이슈에서 Cursor Background Agents 및 AWS Remote SWE Agents와 비교합니다.

## 사용 방법

```
# github.com에서
  - 사용자가 `@agent fix this`로 이슈 #842에 라벨 지정
  - 14분 후 PR #1903 생성됨
  - 내용:
    > null 비교자 항목으로 인한 widget.dedupe()의 NPE 수정.
    > 회귀 테스트 추가: widget_test.go::TestDedupeNullComparator.
    > 커버리지 변화: +0.12%
    > 턴 수: 7  비용: $1.80  추적: langfuse:...
    > 라벨: needs-review
```

## Ship It

`outputs/skill-issue-to-pr.md`가 산출물입니다. 라벨이 지정된 이슈를 검토 가능한 PR로 변환하는 GitHub App + 비동기 클라우드 워커로, 제한된 비용과 범위 지정된 자격 증명을 제공합니다.

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 30개 이슈 통과율 | 엔드-투-엔드 성공 (CI 녹색 + 커버리지 OK) |
| 20 | PR 품질 | Diff 크기, 커버리지 변화량, 스타일 준수 여부 |
| 20 | 해결 이슈당 비용 및 지연 시간 | PR당 $ 및 벽시계 시간 |
| 20 | 안전성 | 범위 지정된 토큰, 리포지토리별 예산, 강제 푸시 금지, 자격 증명 위생 |
| 15 | 운영자 UX | 근거 설명 주석, 재시도 기능, @-멘션 후속 조치 |
| **100** | | |

## 연습 문제

1. "불안정한 테스트 수정" 모드 추가: 라벨 `@agent stabilize-flake TestX`는 샌드박스 내에서 테스트를 50회 실행하고 안정화 최소 변경 사항을 제안합니다.

2. 세 가지 공유 이슈에서 비용 대비 Cursor Background Agents 성능 비교. 어떤 도구가 어떤 영역에서 우수한지 보고합니다.

3. 예산 대시보드 구현: 저장소별/일별 비용, 사용자별 비용. 이상치 발생 시 알림.

4. "드라이 런" 모드 구축: CI 실행 없이 초안 PR을 열어 리뷰어가 저렴하게 계획을 검토할 수 있도록 합니다.

5. 보존 정책 추가: 7일 이상 병합되지 않은 PR 브랜치는 자동으로 삭제됩니다.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| GitHub App | "범위 기반 봇 신원" | 세분화된 권한 + 단기 설치 토큰을 가진 앱 |
| Async cloud agent | "백그라운드 에이전트" | 터미널이 아닌 클라우드 샌드박스에서 실행되는 비대화형 작업자 |
| Environment inference | "Dockerfile 합성" | 언어 및 패키지 관리자 감지, Dockerfile이 없을 경우 생성 |
| Verification | "샌드박스 내 CI" | PR 열기 전 작업자 내에서 전체 테스트 스위트 실행 |
| Coverage delta | "커버리지 보존" | 베이스 브랜치 대비 에이전트 브랜치의 테스트 커버리지 % 변화량 |
| Per-repo budget | "일일 상한" | 디스패처에서 강제하는 달러 및 PR 수 제한 |
| Rationale | "PR 본문 설명" | 변경 사항과 그 이유에 대한 에이전트의 요약; PR 본문에 필수 포함 |

## 추가 자료

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 표준 비동기 클라우드 에이전트 참조
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 참조
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 상용 대체 솔루션
- [OpenAI Codex (cloud)](https://openai.com/codex) — 호스팅 경쟁 제품
- [Google Jules](https://jules.google) — Google의 호스팅 버전
- [Factory Droids](https://www.factory.ai) — 대체 상용 참조
- [GitHub App documentation](https://docs.github.com/en/apps) — 범위 기반 봇 신원
- [Daytona cloud sandboxes](https://daytona.io) — 참조 샌드박스