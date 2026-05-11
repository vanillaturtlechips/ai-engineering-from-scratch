# 캡스톤 15 — 헌법 기반 안전 장치 + 레드팀 평가 환경

> Anthropic의 Constitutional Classifiers, Meta의 Llama Guard 4, Google의 ShieldGemma-2, NVIDIA의 Nemotron 3 Content Safety, 다국어 지원을 위한 X-Guard가 2026년 안전 분류기 스택을 정의했습니다. garak, PyRIT, NVIDIA Aegis, promptfoo는 표준 적대적 평가 도구가 되었습니다. NeMo Guardrails v0.12는 이들을 프로덕션 파이프라인에 통합합니다. 이 캡스톤은 모든 요소를 통합합니다: 대상 애플리케이션 주변의 계층적 안전 장치, 6개 이상의 공격 패밀리를 실행하는 자율 레드팀 에이전트, 측정 가능한 무해성 델타를 생성하는 헌법 기반 자기 비판 실행.

**유형:** 캡스톤  
**언어:** Python (안전 파이프라인, 레드팀), YAML (정책 구성)  
**선수 과목:** 10단계(기초 LLM), 11단계(LLM 엔지니어링), 13단계(도구), 14단계(에이전트), 18단계(윤리, 안전, 정렬)  
**적용 단계:** P10 · P11 · P13 · P14 · P18  
**소요 시간:** 25시간

## 문제

2026년 LLM 안전의 최전선은 분류기 작동 여부(대략적으로는 작동함)가 아니라, 프로덕션 애플리케이션 주변에 이를 올바르게 구성하여 과도한 거부(over-refusing) 없이 또는 명백한 허점 없이 배치하는 방법입니다. **Llama Guard 4**는 영어 정책 위반을 처리합니다. **X-Guard**(132개 언어 지원)는 다국어 탈옥(jailbreak)을 처리합니다. **ShieldGemma-2**는 이미지 기반 프롬프트 인젝션을 탐지합니다. **NVIDIA Nemotron 3 Content Safety**는 엔터프라이즈 범주를 커버합니다. **Anthropic의 Constitutional Classifiers**는 서빙(serving)이 아닌 훈련 중에 사용되는 별도의 접근법입니다.

공격 진화도 중요합니다. **PAIR**와 **TAP**은 탈옥 자동 발견을 수행합니다. **GCG**는 그래디언트 기반 접미사 공격을 실행합니다. 다중 턴(multi-turn) 및 코드 전환(code-switch) 공격은 에이전트 메모리를 악용합니다. 배포된 모든 LLM은 레드 팀 테스트 범위가 필요하며, **garak**과 **PyRIT**이 표준 도구입니다. 또한 문서화된 완화 전략과 **CVSS** 점수 기반 발견 사항이 필요합니다.

대상 애플리케이션(8B 명령어 튜닝 모델 또는 다른 캡스톤의 **RAG** 챗봇 중 하나)을 강화하고, 6개 이상의 공격 패밀리를 실행한 후, 공격 전후의 무해성 측정 결과를 제출해야 합니다.

## 개념

안전 파이프라인은 5계층으로 구성됩니다. **입력 정제**: 제로 너비 문자 제거, base64/rot13 디코딩, 유니코드 정규화. **정책 계층**: NeMo Guardrails v0.12 레일(오프도메인, 유해성, PII 추출). **분류기 게이트**: 입력에 Llama Guard 4, 비영어 입력에 X-Guard, 이미지 입력에 ShieldGemma-2. **모델**: 대상 LLM. **출력 필터**: 출력에 Llama Guard 4, Presidio PII 제거, 해당 시 인용 강제. **HITL 계층**: 고위험으로 플래그된 출력은 Slack 큐로 이동.

레드팀 테스트 범위는 스케줄러에서 실행됩니다. PAIR과 TAP은 자율적으로 탈옥을 발견합니다. GCG는 기울기 기반 접미사 공격을 실행합니다. ASCII / base64 / rot13 인코딩 공격. 다단계 공격(페르소나 채택, 메모리 악용). 코드 전환 공격(영어와 스와힐리어 또는 태국어 혼합). 각 실행은 CVSS 점수와 공개 타임라인이 포함된 구조화된 결과 파일을 생성합니다.

헌법적-자기비판 실행은 학습 시간 개입입니다. 1,000개의 유해 시도 프롬프트를 사용하여 모델이 응답을 초안으로 작성한 후, 작성된 헌법(해를 끼치지 않음 규칙)에 대해 비판하고, 비판 루프로 재학습합니다. 보류된 평가 세트에서 무해성 변화량(델타)을 측정합니다.

## 아키텍처

```
요청 (텍스트 / 이미지 / 다국어)
      |
      v
입력 정제 (제로 너비 문자 제거, 디코딩, 정규화)
      |
      v
NeMo Guardrails v0.12 레일 (도메인 외, 정책)
      |
      v
분류기 게이트:
  Llama Guard 4 (영어)
  X-Guard (다국어, 132개 언어)
  ShieldGemma-2 (이미지 프롬프트)
  Nemotron 3 콘텐츠 안전성 (엔터프라이즈)
      |
      v (허용됨)
대상 LLM
      |
      v
출력 필터: Llama Guard 4 + Presidio PII + 인용 확인
      |
      v
플래그된 출력에 대한 HITL 계층

병렬:
  레드팀 스케줄러
    -> 가락 (클래식 공격)
    -> PyRIT (조직화된 레드팀)
    -> 자율 탈옥 에이전트 (PAIR + TAP)
    -> GCG 접미사 공격
    -> 다국어 / 코드 전환
    -> 다중 턴 페르소나 채택

출력: CVSS 점수 평가 결과 + 공개 일정 + 이전/이후 무해성 델타
```

## 스택

- **안전 분류기**: Llama Guard 4, ShieldGemma-2, NVIDIA Nemotron 3 Content Safety, X-Guard  
- **가드레일 프레임워크**: NeMo Guardrails v0.12 + OPA  
- **레드 팀 드라이버**: garak (NVIDIA), PyRIT (Microsoft Azure), NVIDIA Aegis, promptfoo  
- **탈옥 에이전트**: PAIR (Chao et al., 2023), Tree-of-Attacks (TAP), GCG 접미사  
- **헌법적 훈련**: Anthropic 스타일 자기 비판 루프 + 비판에 대한 SFT(Supervised Fine-Tuning)  
- **PII 제거**: Presidio  
- **대상**: 8B 명령어 튜닝 모델 또는 다른 캡스톤 프로젝트의 RAG 챗봇 중 하나

## 구축 방법

1. **대상 설정.** vLLM에서 8B 인스트럭션 튜닝 모델을 실행하거나(또는 다른 캡스톤의 RAG 챗봇 재사용), 테스트 대상 애플리케이션을 준비합니다.

2. **안전 파이프라인 통합.** 5계층 파이프라인을 대상 모델 주변에 연결합니다. 각 계층이 개별적으로 관찰 가능한지 확인합니다(Langfuse에서 계층별 스팬 확인).

3. **분류기 적용 범위.** Llama Guard 4, X-Guard(다국어), ShieldGemma-2(이미지)를 로드합니다. 소규모 라벨링된 데이터셋에서 각각 실행하여 기준치를 설정합니다.

4. **레드팀 스케줄러.** garak, PyRIT, PAIR 에이전트, TAP 에이전트, GCG 러너, 다중 턴 공격자, 코드 전환 공격자를 예약합니다. 각각 별도의 큐에서 실행됩니다.

5. **공격 세트.** 6가지 공격 유형: (1) PAIR 자동 탈옥, (2) TAP 공격 트리, (3) GCG 기울기 접미사, (4) ASCII/베이스64/rot13 인코딩, (5) 다중 턴 페르소나, (6) 다국어 코드 전환. 각 유형별 성공률을 보고합니다.

6. **헌법적 자기 비판.** 1,000개의 유해 시도 프롬프트를 선별합니다. 각각에 대해 대상 모델이 응답을 초안으로 작성합니다. 비평가 LLM이 "해 끼치지 말 것", "증거 인용", "불법 요청 거부" 등의 헌법 조항에 따라 점수를 매깁니다. 비평가가 이의를 제기한 프롬프트는 재작성되며, 대상 모델은 비평 개선 페어로 미세 조정됩니다. 보유 평가 세트에서 무해성 개선도를 전후 비교 측정합니다.

7. **과도한 거부 측정.** XSTest와 같은 양성 프롬프트 세트에서 오탐률을 추적합니다. 대상 모델은 양성 질문에 대해 여전히 도움이 되어야 합니다.

8. **CVSS 점수화.** 각 성공적인 탈옥에 대해 CVSS 4.0(공격 벡터, 복잡성, 영향)으로 점수화합니다. 공개 타임라인과 완화 계획을 생성합니다.

9. **범위 자동화.** 위 모든 단계는 크론에서 실행되며, 결과는 큐에 기록되고, 과도한 거부 회귀 경고가 Slack으로 전송됩니다.

## 사용 방법

```
$ safety probe --model=target --family=PAIR --budget=50
[attacker]   PAIR 에이전트가 타겟에서 실행 중
[attack]     시도 1/50: 학술 연구로 위장 ... 차단됨
[attack]     시도 2/50: 역할극 호소 ... 차단됨
[attack]     시도 3/50: 연쇄적 사고 유도 ... 성공
[finding]    CVSS 4.8 중간: 타겟에서 역할극 우회 발견
[range]      50회 중 7회 성공 (성공률 14%)
```

## Ship It

`outputs/skill-safety-harness.md`가 산출물입니다. 프로덕션 등급의 계층적 안전 파이프라인 및 재현 가능한 레드팀 범위와 함께 before/after 무해성 델타(delta)를 포함합니다.

| 가중치 | 평가 기준 | 측정 방법 |
|:-:|---|---|
| 25 | 공격 표면 커버리지 | 6개 이상의 공격 패밀리 실행, 2개 이상의 언어 지원 |
| 20 | 진양성/거짓양성 트레이드오프 | 공격 차단률 대 XSTest 무해 통과율 |
| 20 | 자체 평가 델타 | 보류된 평가에서의 before/after 무해성 |
| 20 | 문서화 및 공개 | 타임라인이 포함된 CVSS 점수 평가 결과 |
| 15 | 자동화 및 재현성 | 모든 작업이 크론(cron)으로 실행되고 알림 발생 |
| **100** | | |

## 연습 문제

1. RAG 챗봇에 대해 프롬프트 인젝션용 가라크(garak) 플러그인을 실행하고, 출력 필터 레이어가 있는 경우와 없는 경우의 공격 성공률을 비교하시오.

2. 일곱 번째 공격 패밀리(가족)를 추가하시오: 검색 문서를 통한 간접 프롬프트 인젝션. 추가로 필요한 방어 수단을 측정하시오.

3. "도움 제공 거부(refuse-with-help)" 모드를 구현하시오: 가드레일이 차단할 때, 대상 모델이 완전한 거부 대신 더 안전한 관련 답변을 제공하도록 한다. XSTest 변화량(delta)을 측정하시오.

4. 다국어 커버리지 격차: X-Guard가 성능이 낮은 언어를 찾아 제안하시오. 해당 언어를 대상으로 한 파인튜닝(fine-tuning) 데이터셋을 제안하시오.

5. 30B 모델에서 헌법 기반 자기 비판(constitutional self-critique)을 실행하고, 변화량(delta)이 확장되는지 측정하시오.

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|---------------------|-----------|
| 계층적 안전(Layered safety) | "방어 심층화(Defense in depth)" | 입력, 게이트, 출력, HITL(Human-in-the-Loop)에서의 다중 가드레일 |
| 라마 가드 4(Llama Guard 4) | "메타의 안전 분류기(Meta's safety classifier)" | 2026년 참조 입력/출력 콘텐츠 분류기 |
| 페어(PAIR) | "탈옥 에이전트(Jailbreak agent)" | LLM 기반 탈옥 발견 논문(Chao et al.) |
| 탑(TAP) | "공격 트리(Tree-of-Attacks)" | 페어(PAIR)의 트리 탐색 변형 |
| GCG | "탐욕적 좌표 기울기(Greedy coordinate gradient)" | 기울기 기반 적대적 접미사 공격 |
| 헌법 기반 자기 비판(Constitutional self-critique) | "앤트로픽 스타일 훈련(Anthropic-style training)" | 대상 초안 -> 비평가 점수 -> 재작성 -> 재훈련 |
| XSTest | "양성 프로브 세트(Benign probe set)" | 과도한 거부 회귀 벤치마크 |
| CVSS 4.0 | "심각도 점수(Severity score)" | 안전 발견을 위한 표준 취약점 점수 체계 |

## 추가 자료

- [Anthropic Constitutional Classifiers](https://www.anthropic.com/research/constitutional-classifiers) — 학습 시간 참조
- [Meta Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026년 입력/출력 분류기
- [Google ShieldGemma-2](https://huggingface.co/google/shieldgemma-2b) — 이미지 + 멀티모달 안전성
- [NVIDIA Nemotron 3 Content Safety](https://developer.nvidia.com/blog/building-nvidia-nemotron-3-agents-for-reasoning-multimodal-rag-voice-and-safety/) — 엔터프라이즈 참조
- [X-Guard (arXiv:2504.08848)](https://arxiv.org/abs/2504.08848) — 132개 언어 다국어 안전성
- [garak](https://github.com/NVIDIA/garak) — NVIDIA 레드 팀 툴킷
- [PyRIT](https://github.com/Azure/PyRIT) — Microsoft 레드 팀 프레임워크
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 레일 프레임워크
- [PAIR (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — 탈옥 에이전트 논문