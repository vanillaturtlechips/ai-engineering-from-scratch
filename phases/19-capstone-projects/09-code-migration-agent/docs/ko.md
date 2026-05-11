# Capstone 09 — 코드 마이그레이션 에이전트 (리포지토리 수준 언어/런타임 업그레이드)

> Amazon의 MigrationBench(Java 8에서 17로)와 Google의 App Engine Py2-to-Py3 마이그레이터는 2026년 기준을 설정했습니다. Moderne의 OpenRewrite는 대규모로 결정적 AST 재작성을 수행합니다. Grit은 codemod 스타일 DSL로 동일한 문제를 대상으로 합니다. 프로덕션 패턴은 두 가지를 결합합니다: 안전한 재작성을 위한 결정적 기반과 모호한 경우를 위한 에이전트 계층, 분기별 빌드를 위한 샌드박스, PR 열기 전에 녹색으로 전환되는 테스트 하네스. 캡스톤은 50개의 실제 리포지토리를 마이그레이션하고 실패 분류 체계와 함께 통과율을 게시하는 것입니다.

**유형:** 캡스톤  
**언어:** Python(에이전트), Java/Python(대상), TypeScript(대시보드)  
**선수 조건:** Phase 5(NLP), Phase 7(트랜스포머), Phase 11(LLM 엔지니어링), Phase 13(도구), Phase 14(에이전트), Phase 15(자율성), Phase 17(인프라)  
**적용 단계:** P5 · P7 · P11 · P13 · P14 · P15 · P17  
**소요 시간:** 30시간

## 문제

대규모 코드 마이그레이션은 2026년 코딩 에이전트의 가장 깔끔한 프로덕션 애플리케이션 중 하나입니다. 그라운드 트루스(ground truth)는 명확합니다(마이크레이션 후 테스트 스위트가 통과하는가?), 보상은 실제적입니다(Java-8 플릿 마이그레이션은 인력 규모의 프로젝트입니다), 그리고 벤치마크는 공개되어 있습니다(MigrationBench 50-리포지토리 서브셋). Moderne의 OpenRewrite는 결정론적 측면을 처리합니다. 에이전트 계층은 OpenRewrite 레시피가 처리할 수 없는 모든 것을 처리합니다: 모호한 리라이트, 빌드 시스템 드리프트, 긴 꼬리 문법, 전이적 의존성 파손.

Java 8 리포지토리(또는 Python 2 리포지토리)를 입력으로 받아 녹색 CI(green-CI) 마이그레이션 브랜치를 생성하는 에이전트를 구축할 것입니다. 통과율, 테스트 커버리지 보존, 리포지토리당 비용을 측정하고 실패 분류 체계를 구축할 것입니다. 결정론적 전용 기준선과의 병렬 비교는 에이전트의 실제 가치가 어디에 있는지 알려줍니다.

## 개념

파이프라인은 두 계층으로 구성됩니다. **결정적 기반(deterministic substrate)**(Java용 OpenRewrite, Python용 libcst)은 안전하게 대부분의 기계적 리라이팅(import, 메서드 시그니처, null-안전성 수정, try-with-resources, 사용 중단된 API 대체)을 수행합니다. 이는 빠르며 감사 가능한 diff를 생성합니다. **에이전트 계층**(OpenAI Agents SDK 또는 Claude Opus 4.7 및 GPT-5.4-Codex 기반 LangGraph)은 레시피로 처리할 수 없는 사례를 처리합니다: 빌드 파일 업그레이드(Maven/Gradle/pyproject), 전이적 의존성 충돌, 테스트 불안정성, 사용자 정의 어노테이션.

각 리포지토리는 대상 런타임이 미리 설치된 Daytona 샌드박스를 받습니다. 에이전트는 반복 작업을 수행합니다: 빌드 실행 → 실패 분류 → 수정 적용 → 재실행. 하드 제한: 리포지토리당 30분, $8, 20번의 에이전트 턴. 모든 테스트가 통과하고 커버리지 감소(delta)가 음수(-)가 아니면 브랜치가 PR을 엽니다. 그렇지 않으면 리포지토리는 증거와 함께 실패 분류로 기록됩니다.

실패 분류 체계가 결과물입니다. 50개 리포지토리에서 어떤 문제가 발생했나요? 전이적 의존성? 사용자 정의 어노테이션? 빌드 도구 버전? 마이그레이션과 무관한 테스트 불안정성? 각 분류에는 카운트와 예시 diff가 포함됩니다. 향후 레시피 작성자는 상위 3개 항목을 대상으로 할 수 있습니다.

## 아키텍처

```
타겟 저장소
      |
      v
OpenRewrite / libcst 결정적 레시피
   (안전, 고속, 감사 가능, 수정 사항의 ~70-80%)
      |
      v
브랜치별 Daytona 샌드박스
      |
      v
에이전트 루프 (Claude Opus 4.7 / GPT-5.4-Codex):
   - 빌드 실행 -> 실패 캡처
   - 실패 분류 (빌드, 테스트, 린트)
   - 수정 적용 (패치 또는 레시피 재시도)
   - 재실행
   - 예산: 30분, $8, 20턴
      |
      v
테스트 + 커버리지 델타 게이트
      |
      v (통과)
PR 열기
      |
      v (실패)
실패 분류로 파일링 + 재현 사례 첨부
```

## 스택

- **결정적 기반(Deterministic substrate)**: OpenRewrite (Java) 또는 libcst (Python)  
- **에이전트(Agent)**: OpenAI Agents SDK 또는 LangGraph over Claude Opus 4.7 + GPT-5.4-Codex  
- **샌드박스(Sandbox)**: Daytona devcontainers per branch, 사전 설치된 대상 런타임 (Java 17 / Python 3.12)  
- **빌드 시스템(Build systems)**: Maven, Gradle, uv (Python)  
- **벤치마크(Benchmarks)**: Amazon MigrationBench 50-리포지토리 서브셋 (Java 8 to 17), Google App Engine Py2-to-Py3 리포지토리  
- **테스트 하니스(Test harness)**: 병렬 실행기(parallel runner), 커버리지 도구 (Jacoco (Java) 또는 coverage.py (Python))  
- **관측 가능성(Observability)**: Langfuse + 모든 diff 청크(chunk)별 트레이스 번들(trace bundle)  
- **대시보드(Dashboard)**: 실패 분류(failure-taxonomy) 대시보드, 클래스별 카운트 및 예시 diff 제공

## 빌드하기

1. **레시피 단계.** OpenRewrite (Java) 또는 libcst (Python) 레시피를 먼저 실행합니다. 기계적인 마이그레이션의 70-80%를 포착합니다. "레시피" 커밋으로 커밋합니다.

2. **빌드 시도.** Daytona 샌드박스: 대상 런타임을 설치하고 빌드를 실행합니다. 성공하면 테스트 단계로 건너뜁니다. 실패하면 에이전트에게 전달합니다.

3. **에이전트 루프.** 도구(`run_build`, `read_file`, `edit_file`, `run_test`, `git_diff`)가 있는 LangGraph. 에이전트는 실패 원인(의존성, 구문, 테스트, 빌드 도구)을 분류하고 대상 수정을 적용합니다. 재실행합니다.

4. **예산 제한.** 레포지토리당 30분 벽시계 시간, $8 비용, 20번의 에이전트 턴. 초과 시 현재 diff와 함께 "예산_초과"로 중단 및 기록합니다.

5. **테스트 + 커버리지 게이트.** 빌드가 성공하면 테스트 스위트를 실행합니다. 기본 레포지토리와 커버리지를 비교합니다. 커버리지가 2% 이상 감소하면 "커버리지_회귀"로 기록합니다.

6. **PR 생성.** 성공 시 브랜치를 푸시하고, 적용된 레시피와 에이전트가 작성한 커밋 요약을 포함한 PR을 엽니다.

7. **실패 분류.** 각 실패한 레포지토리에 클래스를 태그합니다: `dep_upgrade_required`, `build_tool_drift`, `custom_annotation`, `test_flake`, `syntax_edge_case`, `budget_exhausted`. 대시보드를 구축합니다.

8. **50개 레포지토리 실행.** MigrationBench 하위 집합에서 실행합니다. 클래스별 통과율, 레포지토리당 비용, 커버리지 유지율, 그리고 결정적 전용 기준과의 비교를 보고합니다.

## 사용 방법

```
$ migrate legacy-java-service --target java17
[레시피]   27개 재작성 적용됨 (JUnit 4->5, HashMap 초기화, try-with-resources)
[빌드]    실패: 심볼 찾을 수 없음 sun.misc.BASE64Encoder
[에이전트]    턴 1 분류: 제거된 JDK API
[에이전트]    턴 2 적용: sun.misc.BASE64Encoder -> java.util.Base64
[빌드]    성공
[테스트]    412/412 통과; 커버리지 84.1% -> 84.3%
[PR]       #1841 생성됨  비용=$3.20  턴=4
```

## Ship It

`outputs/skill-migration-agent.md`가 산출물입니다. 주어진 리포지토리에서 결정적 레시피를 실행한 후 에이전트 루프를 통해 녹색으로 마이그레이션된 브랜치를 생성하거나, 리포지토리를 분류 체계 클래스 아래에 파일링합니다.

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | MigrationBench 통과율 | 50개 리포지토리 부분 집합에서 pass@1 |
| 20 | 테스트 커버리지 보존 | 기준 대비 평균 커버리지 변화량 |
| 20 | 리포지토리당 마이그레이션 비용 | 통과 실행 시 $/리포지토리 |
| 20 | 에이전트/결정적 도구 통합 | OpenRewrite가 처리한 수정 대비 에이전트 작성 수정 비율 |
| 15 | 실패 분석 문서 | 예시와 함께 한 분류 체계 완성도 |
| **100** | | |

## 연습 문제

1. OpenRewrite만 사용하여(에이전트 없이) 마이그레이션 파이프라인을 실행해 보세요. 전체 파이프라인과 통과율을 비교하고, 에이전트 단독 사용이 차이를 만드는 경우를 식별하세요.

2. "lint-clean" 검사 구현: 마이그레이션 후 스타일 린터(Java는 spotless, Python은 ruff)를 실행하세요. 새로운 린트 오류가 발생하면 PR을 실패 처리하세요. 커버리지는 유지되지만 스타일이 퇴보한 비율을 측정하세요.

3. "minimal-diff" 최적화기 추가: 에이전트의 브랜치가 테스트를 통과한 후, 두 번째 패스로 불필요한 변경 사항을 제거하세요. diff 크기 감소율을 보고하세요.

4. 세 번째 마이그레이션으로 확장: Node 18에서 Node 22로. 샌드박스 래핑을 재사용하고, 커스텀 codemod를 위해 레시피 레이어를 교체하세요.

5. UX 지표로 "첫 번째 녹색 빌드까지 소요 시간(TTFGB)"을 측정하세요. 목표: p50 기준 10분 이내.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 결정적 기반(deterministic substrate) | "레시피 엔진" | OpenRewrite / libcst: 안전성 보장이 있는 선언적 AST 변환 |
| 코드모드(codemod) | "코드 수정 프로그램" | 소스 코드를 기계적으로 변경하는 재작성 규칙 |
| 빌드 드리프트(build drift) | "도구 버전 차이" | 메이저 버전 간 미묘한 Maven / Gradle / uv 동작 변화 |
| 실패 분류(failure class) | "분류 체계 버킷" | 리포지토리가 마이그레이션되지 않은 레이블된 이유: 종속성(dep), 구문(syntax), 테스트(test), 빌드 도구(build-tool), 예산(budget) |
| 커버리지 델타(coverage delta) | "커버리지 보존" | 기준 브랜치에서 마이그레이션된 브랜치까지의 테스트 커버리지 % 변화 |
| 에이전트 턴(agent turn) | "도구 호출 라운드" | 에이전트 루프에서 한 번의 계획(plan) -> 실행(act) -> 관찰(observe) 주기 |
| 예산 소진(budget exhaustion) | "한도에 도달" | 리포지토리가 30분 / $8 / 20턴 제한 내에서 통과하지 못한 상태 |

## 추가 자료

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026년 기준 벤치마크
- [Moderne.io OpenRewrite 플랫폼](https://www.moderne.io) — 결정적 기반 참조
- [OpenRewrite 문서](https://docs.openrewrite.org) — 레시피 작성
- [Grit.io](https://www.grit.io) — 대체 코드모드 DSL
- [OpenAI 샌드박싱 마이그레이션 쿡북](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 참조
- [Google App Engine Py2 to Py3 마이그레이션 도구](https://cloud.google.com/appengine) — 대체 마이그레이션 벤치마크
- [libcst](https://github.com/Instagram/LibCST) — Python 결정적 기반
- [Daytona 샌드박스](https://daytona.io) — 브랜치별 샌드박스 참조