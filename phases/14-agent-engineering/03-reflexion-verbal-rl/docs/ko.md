# 리플렉션(Reflexion): 언어 기반 강화 학습

> 기울기 기반 강화 학습(gradient-based RL)은 실패 모드를 수정하기 위해 수천 번의 시행과 GPU 클러스터가 필요합니다. 리플렉션(Shinn et al., NeurIPS 2023)은 자연어로 이를 수행합니다: 실패한 시행 후 에이전트는 반성(reflection)을 작성하고, 이를 에피소드 메모리(episodic memory)에 저장한 다음, 다음 시행을 해당 메모리에 조건화합니다. 이는 레타(Letta)의 수면 시간 계산, 클로드 코드(Claude Code)의 CLAUDE.md 학습 내용, 프로 워크플로우(pro-workflow)의 학습 규칙(learn-rule) 뒤에 있는 패턴입니다.

**유형:** 구현(Build)  
**언어:** Python (표준 라이브러리)  
**사전 요구 사항:** 14단계 · 01 (에이전트 루프), 14단계 · 02 (ReWOO)  
**소요 시간:** ~60분

## 학습 목표

- Reflexion의 세 가지 구성 요소(Actor, Evaluator, Self-Reflector)와 에피소드 메모리(episodic memory)의 역할을 설명할 수 있다.
- 이진 평가기(binary evaluator), 반성 버퍼(reflection buffer), 새로운 재시도(fresh re-attempts)를 활용한 stdlib Reflexion 루프를 구현할 수 있다.
- 주어진 작업에 대해 스칼라(scalar), 휴리스틱(heuristic), 자기 평가(self-evaluated) 피드백 소스 중 적절한 것을 선택할 수 있다.
- 언어적 강화(verbal reinforcement)가 경사 기반 강화 학습(gradient-based RL)이 수천 번의 시행으로 수정해야 하는 오류를 어떻게 포착하는지 설명할 수 있다.

## 문제 정의

에이전트가 작업에 실패한다. 표준 강화 학습(RL)에서는 수천 번의 추가 시행을 실행하고, 기울기(gradient)를 계산하며, 가중치(weights)를 업데이트한다. 이는 비용이 많이 들고 느리며, 대부분의 프로덕션 에이전트는 모든 실패에 대한 훈련 예산을 갖지 못한다.

Reflexion(Shinn et al., arXiv:2303.11366)은 다른 질문을 던진다: 에이전트가 실패 이유를 생각하고 그 생각을 프롬프트에 반영하여 다시 시도하면 어떨까? 가중치 업데이트도, 기울기도 필요 없이 시행 사이에 자연어(natural language)만으로 저장된 정보를 활용한다.

결과: ALFWorld에서 ReAct 및 다른 파인튜닝(fine-tuning)되지 않은 베이스라인들을 능가한다. HotpotQA에서는 ReAct 대비 성능을 개선했다. 코드 생성(HumanEval/MBPP)에서는 당시 최첨단(state of the art) 성능을 기록했다. 모든 것이 단일 기울기(gradient) 단계 없이 달성되었다.

## 개념

### 세 가지 구성 요소

```
Actor         : 궤적 생성 (ReAct 스타일 루프)
Evaluator     : 궤적 평가 — 이진, 휴리스틱, 또는 자가 평가
Self-Reflector: 실패에 대한 자연어 반성문 작성
```

하나의 데이터 구조:

```
에피소드 메모리: 이전 반성문 목록, 다음 시도 프롬프트 앞에 추가
```

한 번의 시도는 Actor를 실행합니다. Evaluator가 점수를 매깁니다. 점수가 낮으면 Self-Reflector가 반성문을 생성합니다("질문을 X에 대해 묻는 것으로 잘못 읽어 도구를 잘못 선택했습니다"). 반성문은 에피소드 메모리에 저장됩니다. 다음 시도는 새로 시작하지만 반성문을 참조합니다.

### 세 가지 평가자 유형

1. **스칼라(Scalar)** — 외부 이진 신호. ALFWorld 성공/실패. HumanEval 테스트 통과/실패. 가장 간단하며 신호 강도가 가장 높음.
2. **휴리스틱(Heuristic)** — 미리 정의된 실패 시그니처. "에이전트가 동일한 액션을 연속으로 두 번 생성하면 '멈춤'으로 표시." "궤적이 50단계를 초과하면 '비효율적'으로 표시."
3. **자가 평가(Self-evaluated)** — LLM이 자신의 궤적을 평가. 정답 데이터가 없을 때 필요. 신호 강도가 약함; 도구 기반 검증(레슨 05 — CRITIC)과 잘 결합됨.

2026년 기본값은 혼합 방식: 사용 가능한 경우 스칼라, 불가능한 경우 자가 평가, 안전 장치로 휴리스틱.

### 일반화 가능한 이유

Reflexion은 새로운 알고리즘이라기보다는 명명된 패턴에 가깝습니다. 거의 모든 프로덕션 "자가 치유" 에이전트가 어떤 변형을 실행합니다:

- Letta의 수면 시간 계산(레슨 08): 별도의 에이전트가 과거 대화를 반성하고 메모리 블록에 기록.
- Claude Code의 `CLAUDE.md` / "메모리 저장" 패턴: 반성문을 학습 내용으로 캡처하여 향후 세션 앞에 추가.
- pro-workflow의 `/learn-rule` 명령어: 수정 사항을 명시적 규칙으로 캡처.
- LangGraph의 반성 노드: 출력을 평가하고 필요 시 수정으로 라우팅하는 노드.

모두 동일한 통찰에서 파생됩니다: 자연어는 실행 간에 "실패로부터 배운 것"을 전달하기에 충분히 풍부한 매체입니다.

### 작동하는 경우와 작동하지 않는 경우

Reflexion이 작동하는 경우:

- 명확한 실패 신호가 있는 경우(테스트 실패, 도구 오류, 오답).
- 작업 클래스가 재현 가능한 경우(동일한 유형의 질문을 다시 할 수 있음).
- 반성문이 궤적을 개선할 여지가 있는 경우(충분한 액션 예산).

Reflexion이 도움이 되지 않는 경우:

- 에이전트가 첫 번째 시도에서 이미 성공한 경우.
- 실패가 외부적인 경우(네트워크 다운, 도구 고장) — "네트워크가 다운되었다"는 반성문은 향후 실행에 도움이 되지 않음.
- 반성문이 미신으로 변하는 경우 — 일회성 불안정한 실행에 대한 서술을 저장하는 경우.

2026년 함정: 메모리 부패. 반성문이 축적되면 일부는 구식이 되거나 잘못됨; 에피소드 버퍼가 커질수록 재실행 속도가 느려짐. 완화 방법: 주기적 압축(레슨 06), 반성문 TTL 설정, 또는 별도의 수면 시간 정리 에이전트(Letta).

## 빌드하기

`code/main.py`는 장난감 퍼즐에 대한 Reflexion을 구현합니다: 목표값에 합산되는 3-요소 리스트를 생성합니다. Actor는 후보 리스트를 생성하고, Evaluator는 합을 확인하며, Self-Reflector는 실패 원인에 대한 한 줄 설명을 작성합니다. 이 반성 내용은 다음 시도를 위한 에피소드 메모리에 저장됩니다.

구성 요소:

- `Actor` — 반성 내용을 볼 때 개선되는 스크립트 정책입니다.
- `Evaluator.binary()` — 목표 합산에 대한 통과/실패를 평가합니다.
- `SelfReflector` — 실패에 대한 한 줄 진단을 생성합니다.
- `EpisodicMemory` — TTL(시간-유효성) 시맨틱스를 가진 유계 리스트입니다.

실행 방법:

```
python3 code/main.py
```

트레이스에는 3번의 시도가 표시됩니다. 1차 시도는 실패하고 반성 내용이 저장되며, 2차 시도는 반성 내용을 보고 개선되지만 여전히 실패합니다. 3차 시도는 성공합니다. 기준 실행(반사 없음)과 비교하면 1차 시도의 답변에 계속 머무르게 됩니다.

## 사용 방법

LangGraph는 리플렉션(reflection)을 노드 패턴으로 제공합니다. Claude Code의 `/memory` 명령어와 pro-workflow의 `/learn-rule`은 에피소드 버퍼를 마크다운 파일로 외부화합니다. Letta의 수면 시간 계산은 다운타임 동안 Self-Reflector를 실행하여 주 에이전트가 지연 시간(latency)에 구속되도록 합니다. OpenAI Agents SDK는 리플렉션(Reflexion)을 직접 제공하지 않으며, 점수(score)로 궤적(trajectory)을 거부하는 커스텀 가드레일(Guardrail)과 실행 간 지속되는 메모리 `Session`으로 직접 구축해야 합니다.

## Ship It

`outputs/skill-reflexion-buffer.md`는 반사(reflection) 캡처, TTL(Time-To-Live), 중복 제거 기능을 갖춘 에피소드 버퍼를 생성 및 관리합니다. 작업 클래스(task class)와 실패 사례를 기반으로, 다음 시도에 실제로 도움이 되는 반사(일반적인 "더 주의하라"와 같은 내용이 아닌)를 출력합니다.

## 연습 문제

1. 이진 평가자에서 거리 메트릭(목표까지의 거리)을 반환하는 스칼라 평가자(scalar evaluator)로 전환해 보세요. 수렴 속도가 더 빨라지나요?
2. 리플렉션(reflections)에 10회 시행(TTL)을 추가해 보세요. 해당 시점 이후에 오래된 리플렉션이 성능 향상에 도움이 되나요, 아니면 방해가 되나요?
3. 휴리스틱 평가자(heuristic evaluator) 구현: 동일한 액션이 반복되면 시행(trial)을 "멈춤(stuck)" 상태로 표시합니다. 이것이 Self-Reflector와 어떻게 상호작용하나요?
4. 리플렉션을 무시하는 적대적 액터(adversarial Actor)로 Reflexion을 실행해 보세요. 액터가 리플렉션을 인지하도록 강제하는 최소한의 리플렉션 프롬프트 엔지니어링은 무엇인가요?
5. Reflexion 논문의 AlfWorld 섹션 4를 읽어보세요. 130% 성공률 향상 개념을 개념적으로 재현해 보세요: 순수 ReAct 대비 핵심 차이점은 무엇인가요?

## 주요 용어

| 용어 | 사람들이 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| 리플렉션(Reflexion) | "자기 수정(Self-correction)" | Shinn et al. 2023 — 액터(Actor), 평가자(Evaluator), 자기 반성자(Self-Reflector) 및 에피소드 메모리(episodic memory) |
| 언어적 강화(Verbal reinforcement) | "경사 하강 없이 학습(Learning without gradients)" | 다음 시행의 프롬프트 앞에 추가되는 자연어 기반 반성 |
| 에피소드 메모리(Episodic memory) | "작업별 반성(Per-task reflections)" | 하나의 작업 클래스에 대한 이전 반성 기록을 저장하는 제한된 버퍼 |
| 스칼라 평가자(Scalar evaluator) | "이진 성공 신호(Binary success signal)" | 정답(ground truth) 기반의 성공/실패 또는 수치 점수 |
| 휴리스틱 평가자(Heuristic evaluator) | "패턴 기반 감지기(Pattern-based detector)" | 미리 정의된 실패 시그니처(예: 무한 루프, 과도한 단계 수) |
| 자기 평가자(Self-evaluator) | "LLM이 자신의 추적을 평가하는 것(LLM-as-judge on own trace)" | 정답(ground truth)이 없을 때 신호 강도가 낮은 대체 수단 — 도구 기반 검증과 함께 사용 |
| 메모리 열화(Memory rot) | "오래된 반성(Stale reflections)" | 에피소드 버퍼가 더 이상 유효하지 않은 항목으로 채워짐; 압축(compaction) 또는 TTL(Time-To-Live)로 해결 |
| 수면 시간 반성(Sleep-time reflection) | "비동기 자기 반성(Async self-reflection)" | 핫 경로(hot path) 외부에서 자기 반성자(Self-Reflector)를 실행하여 주 에이전트의 속도 유지 |

## 추가 자료

- [Shinn et al., Reflexion: 언어 에이전트와 언어적 강화 학습 (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — 표준 논문
- [Letta, 수면 시간 컴퓨팅](https://www.letta.com/blog/sleep-time-compute) — 프로덕션 환경에서의 비동기 반성
- [Anthropic, AI 에이전트를 위한 효과적인 컨텍스트 엔지니어링](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — 컨텍스트 일부로서의 에피소드 버퍼 관리
- [LangGraph 개요](https://docs.langchain.com/oss/python/langgraph/overview) — 반성 노드 패턴