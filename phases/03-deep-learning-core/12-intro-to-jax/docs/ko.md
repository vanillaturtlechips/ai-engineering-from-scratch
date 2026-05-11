# JAX 소개

> PyTorch는 텐서를 변형합니다. TensorFlow는 그래프를 구축합니다. JAX는 순수 함수를 컴파일합니다. 마지막 방식은 딥러닝에 대한 사고방식을 바꿉니다.

**유형:** 구축(Build)  
**언어:** Python  
**선수 지식:** Phase 03 Lessons 01-10, 기본 NumPy  
**소요 시간:** ~90분

## 학습 목표

- JAX의 함수형 API(`jax.numpy`, `jax.grad`, `jax.jit`, `jax.vmap`)를 사용하여 순수 함수형 신경망 코드 작성
- PyTorch의 즉시 실행(eager execution) 기반 가변성(mutation)과 JAX의 함수형 컴파일 모델의 주요 설계 차이점 설명
- `jit` 컴파일과 `vmap` 벡터화를 적용하여 순진한(naive) Python 대비 훈련 루프 가속화
- JAX에서 간단한 네트워크 훈련 및 PyTorch의 객체 지향 접근법과 비교한 명시적 상태 관리(contrast) 수행

## 문제

당신은 PyTorch에서 신경망을 구축하는 방법을 알고 있습니다. `nn.Module`을 정의하고 `.backward()`를 호출한 후 옵티마이저를 스텝합니다. 작동합니다. 수백만 명의 사람들이 이를 사용합니다.

하지만 PyTorch에는 DNA에 내재된 제약 조건이 있습니다: Python에서 작업을 즉시(eagerly) 하나씩 추적합니다. 모든 `tensor + tensor` 연산은 별도의 커널 실행입니다. 모든 훈련 단계는 동일한 Python 코드를 재해석합니다. 이는 5400억 개의 매개변수를 가진 모델을 2,048개의 TPU에서 훈련시켜야 할 때까지는 잘 작동합니다. 그러면 오버헤드가 치명적이 됩니다.

Google DeepMind는 Gemini를 JAX로 훈련시켰습니다. Anthropic은 Claude를 JAX로 훈련시켰습니다. 이들은 소규모 작업이 아닙니다 — 지구상에서 가장 큰 신경망 훈련 실행 사례입니다. 그들은 훈련 루프를 Python 호출 시퀀스가 아닌 컴파일 가능한 프로그램으로 처리하기 때문에 JAX를 선택했습니다.

JAX는 NumPy에 세 가지 초능력(automatic differentiation, XLA로의 JIT 컴파일, 자동 벡터화)을 더한 것입니다. 하나의 예제를 처리하는 함수를 작성하면, JAX는 배치 처리, 그래디언트 계산, 기계 코드 컴파일, 여러 장치에서의 실행을 지원하는 함수를 제공합니다. 원본 함수를 변경할 필요 없이 모든 것이 가능합니다.

## 개념

### JAX 철학

JAX는 함수형 프레임워크입니다. 클래스도, 가변 상태도, `.backward()` 메서드도 없습니다. 대신:

| PyTorch | JAX |
|---------|-----|
| 상태를 가진 `nn.Module` 클래스 | 순수 함수: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 즉시 실행(Eager execution) | XLA를 통한 JIT 컴파일 |
| `for x in batch:` 수동 루프 | `jax.vmap(f)` 자동 벡터화 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 자동 병렬화 |
| 가변 `model.parameters()` | 불변 pytree 배열 |

이것은 스타일 선호가 아닙니다. 컴파일러 제약입니다. JIT 컴파일은 순수 함수를 요구합니다 — 동일한 입력은 항상 동일한 출력을 생성하며, 부작용이 없습니다. 이 제약이 100배 속도 향상을 가능하게 합니다.

### jax.numpy: 익숙한 표면

JAX는 NumPy API를 가속기에서 재구현합니다:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

동일한 함수 이름. 동일한 브로드캐스팅 규칙. 동일한 슬라이싱 의미. 하지만 배열은 GPU/TPU에 있으며, 모든 연산은 컴파일러에 의해 추적 가능합니다.

중요한 차이점 하나: JAX 배열은 불변입니다. `a[0] = 5`는 불가능합니다. 대신: `a = a.at[0].set(5)`. 일주일 동안은 어색하지만, 곧 이해됩니다 — 불변성이 `grad`, `jit`, `vmap` 같은 변환을 조합 가능하게 합니다.

### jax.grad: 함수형 자동 미분

PyTorch는 텐서에 그래디언트를 첨부합니다(`.grad`). JAX는 함수에 그래디언트를 첨부합니다.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`는 함수를 받아 그래디언트를 계산하는 새로운 함수를 반환합니다. `.backward()` 호출도, 텐서에 저장된 계산 그래프도 없습니다. 그래디언트는 호출, 조합, JIT-컴파일 가능한 또 다른 함수일 뿐입니다.

이것은 임의로 조합 가능합니다:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

2차 도함수. 3차 도함수. 야코비안. 헤시안. 모두 `grad`를 조합하여 가능합니다. PyTorch도 이 기능을 지원하지만(`torch.autograd.functional.hessian`), 이는 추가 기능입니다. JAX에서는 기반입니다.

제약: `grad`는 순수 함수에서만 작동합니다. 내부에 `print` 문도 (실행이 아닌 추적 중에 실행됨), 외부 상태 변경도, 명시적 키 관리 없는 난수 생성도 불가능합니다.

### jit: XLA로 컴파일

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

첫 호출 시, JAX는 함수를 추적합니다 — 연산을 실행하지 않고 어떤 연산이 발생하는지 기록합니다. 그런 다음 이 추적을 XLA(Accelerated Linear Algebra)에 전달합니다. XLA는 연산을 융합하고, 불필요한 메모리 복사를 제거하며, 최적화된 기계 코드를 생성합니다.

이후 호출은 Python을 완전히 건너뜁니다. 컴파일된 코드는 C++ 속도로 가속기에서 실행됩니다.

JIT가 도움이 되는 경우:
- 훈련 단계 (동일한 계산이 수천 번 반복됨)
- 추론 (동일한 모델, 다른 입력)
- 유사한 형태의 입력으로 여러 번 호출되는 모든 함수

JIT가 해로운 경우:
- 값에 의존하는 Python 제어 흐름 (`if x > 0`에서 x가 추적된 배열인 경우)
- 일회성 계산 (컴파일 오버헤드가 실행 시간을 초과함)
- 디버깅 (추적이 실제 실행을 가림)

제어 흐름 제한은 실제입니다. `jax.lax.cond`는 `if/else`를 대체합니다. `jax.lax.scan`은 `for` 루프를 대체합니다. 이들은 선택 사항이 아닙니다 — 컴파일의 대가입니다.

### vmap: 자동 벡터화

하나의 예제를 처리하는 함수를 작성합니다:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`은 이를 배치로 확장합니다:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`는 `params`는 배치하지 않음(공유), `x`의 축 0을 따라 배치함을 의미합니다. 수동 `for` 루프도, 리쉐이핑도, 배치 차원 스레드도 필요 없습니다. JAX가 배치 차원을 파악하고 전체 계산을 벡터화합니다.

이것은 단순한 구문 설탕이 아닙니다. `vmap`은 Python 루프보다 10-100배 빠르게 실행되는 융합된 벡터화 코드를 생성합니다. 그리고 `jit` 및 `grad`와 조합됩니다:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

예제별 그래디언트. 한 줄입니다. PyTorch에서는 해킹 없이는 거의 불가능합니다.

### pmap: 장치 간 데이터 병렬화

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`은 사용 가능한 모든 장치(GPU/TPU)에 함수를 복제하고 배치를 분할합니다. 함수 내부에서 `jax.lax.pmean`과 `jax.lax.psum`은 장치 간 그래디언트를 동기화합니다.

Google은 `pmap`(및 후속 `shard_map`)을 사용하여 수천 개의 TPU v5e 칩에서 Gemini를 훈련시킵니다. 프로그래밍 모델: 단일 장치 버전을 작성한 후 `pmap`으로 감싸면 완료됩니다.

### Pytrees: 범용 데이터 구조

JAX는 "pytree"에서 작동합니다 — 리스트, 튜플, 딕셔너리, 배열의 중첩된 조합. 모델 파라미터는 pytree입니다:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

모든 JAX 변환 — `grad`, `jit`, `vmap` — 은 pytree를 순회하는 방법을 알고 있습니다. `jax.tree.map(f, tree)`는 모든 리프에 `f`를 적용합니다. 이것이 옵티마이저가 모든 파라미터를 한 번에 업데이트하는 방법입니다:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

`.parameters()` 메서드도, 파라미터 등록도 없습니다. 트리 구조가 모델입니다.

### 함수형 vs 객체 지향

PyTorch는 객체 내부에 상태를 저장합니다:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX는 명시적 상태를 가진 순수 함수를 사용합니다:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params는 전달됩니다. 아무것도 저장되지 않습니다. 아무것도 변경되지 않습니다. 이는 모든 함수를 테스트 가능, 조합 가능, 컴파일 가능하게 합니다. 또한 params를 직접 관리해야 함을 의미합니다 — 또는 Flax나 Equinox 같은 라이브러리를 사용합니다.

### JAX 생태계

JAX는 기본 요소를 제공합니다. 라이브러리는 편의성을 제공합니다:

| 라이브러리 | 역할 | 스타일 |
|---------|------|-------|
| **Flax** (Google) | 신경망 레이어 | 명시적 상태를 가진 `nn.Module` |
| **Equinox** (Patrick Kidger) | 신경망 레이어 | Pytree 기반, Pythonic |
| **Optax** (DeepMind) | 옵티마이저 + 학습률 스케줄 | 조합 가능한 그래디언트 변환 |
| **Orbax** (Google) | 체크포인팅 | pytree 저장/복원 |
| **CLU** (Google) | 메트릭 + 로깅 | 훈련 루프 유틸리티 |

Optax는 표준 옵티마이저 라이브러리입니다. 그래디언트 변환(Adam, SGD, 클리핑)과 파라미터 업데이트를 분리하여 조합을 쉽게 합니다:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### JAX vs PyTorch 사용 시기

| 요소 | JAX | PyTorch |
|--------|-----|---------|
| TPU 지원 | 1급 (Google이 둘 다 구축) | 커뮤니티 유지 관리 (torch_xla) |
| GPU 지원 | 좋음 (XLA를 통한 CUDA) | 최고 수준 (네이티브 CUDA) |
| 디버깅 | 어려움 (추적 + 컴파일) | 쉬움 (즉시 실행, 라인별) |
| 생태계 | 연구 중심 (Flax, Equinox) | 거대 (HuggingFace, torchvision 등) |
| 채용 | 니치 (Google/DeepMind/Anthropic) | 주류 (모든 곳) |
| 대규모 훈련 | 우수 (XLA, pmap, 메시) | 좋음 (FSDP, DeepSpeed) |
| 프로토타이핑 속도 | 느림 (함수형 오버헤드) | 빠름 (변경 및 진행) |
| 프로덕션 추론 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 사용자 | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

솔직한 답변: JAX를 사용할 특정 이유가 없다면 PyTorch를 사용하세요. 그 이유는 — TPU 접근, 예제별 그래디언트 필요, 대규모 다중 장치 훈련, 또는 Google/DeepMind/Anthropic에서 작업하는 경우입니다.

### JAX의 난수

JAX는 전역 난수 상태를 가지지 않습니다. 모든 난수 연산에는 명시적 PRNG 키가 필요합니다:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

처음에는 귀찮습니다. 하지만 장치 및 컴파일 전반에 걸쳐 재현성을 보장합니다 — PyTorch의 `torch.manual_seed`가 멀티 GPU 설정에서 보장할 수 없는 속성입니다.

## 구축

### 단계 1: 설정 및 데이터

JAX와 Optax를 사용하여 MNIST 데이터셋에 3층 MLP를 훈련시킵니다. 784개의 입력, 256개와 128개 뉴런을 가진 두 개의 은닉층, 10개의 출력 클래스입니다.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 단계 2: 파라미터 초기화

클래스 없이, pytree를 반환하는 함수만 있습니다:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He 초기화(He-initialization)를 수동으로 구현했습니다. 하나의 시드에서 분할된 세 개의 PRNG 키를 사용합니다. 모든 가중치는 중첩된 딕셔너리 안의 불변 배열입니다.

### 단계 3: 순전파

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

순수 함수입니다. 파라미터를 입력으로 받아 예측을 출력합니다. `self`도 없고 저장된 상태도 없습니다. `loss_fn`은 소프트맥스(softmax), 로그(log), 음의 평균(negative mean)을 사용하여 교차 엔트로피(cross-entropy)를 처음부터 계산합니다.

### 단계 4: JIT 컴파일된 훈련 단계

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`는 손실 값과 기울기(gradient)를 한 번의 패스로 반환합니다. `@jax.jit` 데코레이터는 두 함수를 XLA로 컴파일합니다. 첫 번째 호출 이후에는 각 훈련 단계가 Python을 건드리지 않고 실행됩니다.

### 단계 5: 훈련 루프

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 에폭(epoch) 동안 훈련합니다. 테스트 정확도는 약 97%입니다. 첫 번째 에폭은 느립니다(JIT 컴파일 때문). 2~10번째 에폭은 빠릅니다.

`.zero_grad()`, `.backward()`, `.step()`이 없는 것에 주목하세요. 전체 업데이트는 하나의 합성된 함수 호출입니다. 기울기 계산, Adam에 의한 변환, 파라미터 적용은 모두 `train_step` 내에서 이루어집니다.

## 사용 방법

### Flax: Google 표준

Flax는 가장 일반적인 JAX 신경망 라이브러리입니다. `nn.Module`을 다시 추가하지만, 명시적인 상태 관리를 제공합니다:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

PyTorch와 동일한 구조이지만, `params`는 모델과 분리되어 있습니다. `model.init()`은 파라미터를 생성합니다. `model.apply(params, x)`는 순전파를 실행합니다. 모델 객체에는 상태가 없습니다.

### Equinox: 파이썬 스타일 대안

Equinox(Patrick Kidger 제작)는 모델을 pytree로 표현합니다:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

모델 자체가 pytree입니다. `.apply()`가 필요 없습니다. 파라미터는 모델의 리프 노드일 뿐입니다. 이는 JAX의 사고 방식에 더 가깝습니다.

### Optax: 조합 가능한 옵티마이저

Optax는 그래디언트 변환과 업데이트를 분리합니다:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

그래디언트 클리핑, 학습률 워밍업, 가중치 감소 — 모두 변환 체인으로 조합됩니다. 각 변환은 그래디언트를 확인하고 수정한 후 다음 단계로 전달합니다. 단일 옵티마이저 클래스가 없습니다.

## Ship It

**설치:**

```bash
pip install jax jaxlib optax flax
```

GPU 지원:

```bash
pip install jax[cuda12]
```

TPU(Google Cloud)용:

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**성능 주의 사항:**

- 첫 JIT 호출은 느림(컴파일). 벤치마킹 전 워밍업 필요.
- JIT 내부에서 JAX 배열에 대한 Python 루프 사용 금지. `jax.lax.scan` 또는 `jax.lax.fori_loop` 사용.
- `jax.debug.print()`는 JIT 내부에서 작동. 일반 `print()`는 작동하지 않음.
- `jax.profiler` 또는 TensorBoard로 프로파일링. XLA 컴파일이 병목 현상을 숨길 수 있음.
- JAX는 기본적으로 GPU 메모리의 75%를 사전 할당. 비활성화하려면 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 설정.

**체크포인팅:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**이 강의는 다음을 생성합니다:**
- `outputs/prompt-jax-optimizer.md` -- 적절한 JAX 옵티마이저(optimizer) 구성 선택용 프롬프트
- `outputs/skill-jax-patterns.md` -- JAX의 함수형 패턴(functional patterns)을 다루는 기술 문서

## 연습 문제

1. MLP에 드롭아웃을 추가하세요. JAX에서 드롭아웃은 PRNG 키가 필요합니다 — 순전파 과정에서 키를 전달하고 각 드롭아웃 레이어마다 키를 분할하세요. 테스트 정확도를 드롭아웃 적용 전후와 비교하세요.

2. `jax.vmap`을 사용하여 32개의 MNIST 이미지 배치에 대한 개별 예제 기울기를 계산하세요. 각 예제에 대한 기울기 노름(norm)을 계산하세요. 어떤 예제에서 기울기가 가장 큰지, 그 이유는 무엇인지 분석하세요.

3. 수동 순전파 함수를 임의의 레이어 수에 대해 작동하는 일반적인 `mlp_forward(params, x)`로 대체하세요. `jax.tree.leaves`를 사용하여 깊이를 자동으로 결정하세요.

4. `@jax.jit`를 사용한 경우와 사용하지 않은 경우의 훈련 단계를 벤치마킹하세요. 각각 100단계를 측정하고 시간 차이를 비교하세요. 하드웨어에서 속도 향상 정도는 얼마나 되며, 첫 호출 시 컴파일 오버헤드는 어떻게 되는지 확인하세요.

5. `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`를 조합하여 기울기 클리핑을 구현하세요. 클리핑 적용 전후로 훈련을 진행하고, 훈련 중 기울기 노름을 플롯하여 효과를 확인하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| XLA | "JAX를 빠르게 만드는 것" | 가속 선형 대수(Accelerated Linear Algebra) — 연산 그래프를 기반으로 연산을 융합하고 최적화된 GPU/TPU 커널을 생성하는 컴파일러 |
| JIT | "Just-in-time 컴파일" | JAX는 첫 호출 시 함수를 추적(trace)하고 XLA로 컴파일한 후, 이후 호출에서 컴파일된 버전을 실행 |
| 순수 함수(Pure function) | "부작용 없음" | 출력이 입력에만 의존하는 함수 — 전역 상태, 변형(mutation), 명시적 키 없는 무작위성 없음 |
| vmap | "자동 배치 처리" | 하나의 예제를 처리하는 함수를 재작성 없이 배치 처리 가능한 함수로 변환 |
| pmap | "자동 병렬화" | 여러 장치에 함수를 복제하고 입력 배치를 분할 |
| 파이트리(Pytree) | "중첩된 딕셔너리 구조" | JAX가 탐색 및 변환할 수 있는 리스트, 튜플, 딕셔너리, 배열의 중첩 구조 |
| 추적(Tracing) | "계산 기록" | JAX는 추상 값으로 함수를 실행하여 실제 결과 계산 없이 계산 그래프를 구축 |
| 함수형 자동 미분(Functional autodiff) | "함수의 grad" | 텐서에 그래디언트 저장소를 첨부하는 것이 아닌 함수 변환을 통해 미분 계산 |
| Optax | "JAX의 옵티마이저 라이브러리" | Adam, SGD, 클리핑, 스케줄링 등 조합 가능한 그래디언트 변환 라이브러리 |
| Flax | "JAX의 nn.Module" | 상태를 명시적으로 유지하면서 레이어 추상화를 추가하는 Google의 JAX용 신경망 라이브러리

## 추가 자료

- JAX 문서: https://jax.readthedocs.io/ -- 공식 문서, `grad`, `jit`, `vmap`에 대한 훌륭한 튜토리얼 포함
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 설계 철학을 설명한 원본 논문
- Flax 문서: https://flax.readthedocs.io/ -- JAX용 Google의 신경망 라이브러리
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax의 Pythonic 대안
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 표준 옵티마이저 라이브러리
- "You Don't Know JAX" (Colin Raffel, 2020) -- T5 저자 중 한 명이 작성한 JAX의 주의 사항과 패턴에 대한 실용적인 가이드