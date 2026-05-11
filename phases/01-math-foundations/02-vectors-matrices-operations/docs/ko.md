# 벡터, 행렬 & 연산

> 모든 신경망은 추가 단계가 포함된 행렬 곱셈에 불과합니다.

**유형:** 빌드  
**언어:** Python, Julia  
**선수 지식:** 1단계, 레슨 01 (선형 대수 직관)  
**소요 시간:** ~60분

## 학습 목표

- 요소별 연산, 행렬 곱셈, 전치, 행렬식, 역행렬을 지원하는 `Matrix` 클래스 구현
- 요소별 곱셈(element-wise multiplication)과 행렬 곱셈(matrix multiplication)의 차이 구분 및 각 적용 사례 설명
- 직접 구현한 `Matrix` 클래스만을 사용하여 단일 밀집 신경망 계층(`relu(W @ x + b)`) 구현
- 브로드캐스팅 규칙(broadcasting rules) 설명 및 신경망 프레임워크에서 편향(bias) 추가 방식 이해

> **참고**:  
> - `W @ x`는 행렬 곱셈을 의미하며, `@` 연산자는 NumPy/PyTorch에서 행렬 곱셈에 사용됩니다.  
> - `relu`는 활성화 함수(ReLU: Rectified Linear Unit)로, 음수 값을 0으로 변환하고 양수 값은 그대로 유지합니다.  
> - `b`는 편향 벡터(bias vector)로, 각 뉴런에 추가되는 상수 값입니다.

## 문제 정의

신경망 모델을 구축하려고 합니다. 코드를 읽다가 다음 내용을 발견했습니다:

```
output = activation(weights @ input + bias)
```

여기서 `@`는 행렬 곱셈을 의미합니다. `weights`는 행렬이고, `input`은 벡터입니다. 만약 이 연산들이 무엇을 하는지 모른다면 이 코드는 마법처럼 보일 것입니다. 하지만 알고 있다면, 이 코드는 레이어의 순전파(forward pass) 과정을 단 세 가지 연산으로 표현한 것입니다.

모델이 처리하는 모든 이미지는 픽셀 값의 행렬로 표현됩니다. 모든 단어 임베딩은 벡터입니다. 모든 신경망의 레이어는 행렬 변환입니다. 변수 없이 코드를 작성할 수 없는 것처럼, 행렬 연산에 능숙하지 않으면 AI 시스템을 구축할 수 없습니다.

이 강의는 행렬 연산에 대한 숙련도를 처음부터 쌓아 올립니다.

## 개념

### 벡터: 순서가 있는 숫자 목록

벡터는 방향과 크기를 가진 숫자 목록입니다. AI에서 벡터는 데이터 포인트, 특징 또는 매개변수를 표현합니다.

```
v = [3, 4]        -- 2D 벡터
w = [1, 0, -2]    -- 3D 벡터
```

2D 벡터 `[3, 4]`는 평면에서 좌표 (3, 4)를 가리킵니다. 이 벡터의 길이(크기)는 5입니다(3-4-5 삼각형).

### 행렬: 숫자 격자

행렬은 2D 격자입니다. 행과 열로 구성됩니다. m x n 행렬은 m개의 행과 n개의 열을 가집니다.

```
A = | 1  2  3 |     -- 2x3 행렬 (2행, 3열)
    | 4  5  6 |
```

신경망에서 가중치 행렬은 입력 벡터를 출력 벡터로 변환합니다. 784개의 입력과 128개의 출력을 가진 레이어는 128x784 가중치 행렬을 사용합니다.

### 형태(shape)의 중요성

행렬 곱셈에는 엄격한 규칙이 있습니다: `(m x n) @ (n x p) = (m x p)`. 내부 차원이 일치해야 합니다.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  가중치        입력        출력

내부 차원: 784 = 784  -- 유효
```

PyTorch에서 형태 불일치 오류가 발생하면 이 때문입니다.

### 연산 맵

| 연산 | 기능 | 신경망 활용 |
|-----------|-------------|-------------------|
| 덧셈 | 요소별 결합 | 출력에 편향 추가 |
| 스칼라 곱셈 | 모든 요소 스케일링 | 학습률 * 기울기 |
| 행렬 곱셈 | 벡터 변환 | 레이어 순전파 |
| 전치 | 행과 열 뒤집기 | 역전파 |
| 행렬식 | 단일 숫자 요약 | 역변환 가능성 확인 |
| 역행렬 | 변환 취소 | 선형 시스템 해결 |
| 항등 행렬 | 아무 작업 안 함 | 초기화, 잔차 연결 |

### 요소별 vs 행렬 곱셈

이 차이는 초보자를 자주 혼란스럽게 합니다.

요소별: 대응하는 위치끼리 곱합니다. 두 행렬의 형태가 같아야 합니다.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

행렬 곱셈: 행과 열의 내적입니다. 내부 차원이 일치해야 합니다.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

다른 연산, 다른 결과, 다른 규칙.

### 브로드캐스팅

출력 행렬에 편향 벡터를 더할 때 형태가 일치하지 않습니다. 브로드캐스팅은 작은 배열을 늘려 맞춥니다.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

브로드캐스팅은 벡터를 행 방향으로 늘립니다:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

모든 현대 프레임워크는 이를 자동으로 수행합니다. 형태가 맞지 않아 보이지만 코드가 실행되는 경우 이해하면 혼란을 방지할 수 있습니다.

## 구축 방법

### 1단계: 벡터 클래스

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 2단계: 핵심 연산을 포함한 행렬 클래스

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 3단계: 작동 확인

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 4단계: 신경망 연결

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

이것은 단일 밀집층입니다: `output = relu(W @ x + b)`. 모든 신경망의 모든 밀집층은 정확히 이 연산을 수행합니다.

## 사용 방법

NumPy는 위의 모든 작업을 더 적은 코드 라인으로 훨씬 빠르게 수행합니다.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (요소별 곱셈) =\n", A * B)
print("A @ B (행렬 곱셈) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\n신경망 레이어: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"출력:\n{output}")
```

Python의 `@` 연산자는 `__matmul__`을 호출합니다. NumPy는 C와 Fortran으로 작성된 최적화된 BLAS 루틴으로 이를 구현합니다. 동일한 수학 연산이지만 100배 더 빠릅니다.

NumPy의 브로드캐스팅:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy는 1D bias를 두 행에 자동으로 브로드캐스팅합니다. 이는 모든 신경망 프레임워크에서 바이어스 추가가 작동하는 방식입니다.

## Ship It

이 레슨은 기하학적 직관을 통해 행렬 연산을 가르치는 프롬프트를 생성합니다. `outputs/prompt-matrix-operations.md`를 참조하세요.

여기서 구축한 Matrix 클래스는 Phase 3, Lesson 10에서 구축하는 미니 신경망 프레임워크의 기반이 됩니다.

## 연습 문제

1. **역행렬 검증.** `A @ A.inverse_2x2()`를 계산하고 단위 행렬이 나오는지 확인하세요. 서로 다른 2×2 행렬 3개로 시도해 보세요. 행렬식(determinant)이 0일 때 어떤 현상이 발생하나요?

2. **3×3 역행렬 구현.** Matrix 클래스를 확장하여 수반 행렬(adjugate) 방법으로 3×3 행렬의 역행렬을 계산하세요. NumPy의 `np.linalg.inv`와 결과를 비교해 테스트하세요.

3. **2층 신경망 구축.** Matrix 클래스만 사용하여(NumPy 미사용) 2층 신경망을 만드세요: 입력(3) -> 은닉층(4) -> 출력(2). 무작위 가중치(weights)를 초기화하고 순전파(forward pass)를 실행한 후 모든 차원(shape)이 올바른지 확인하세요.

## 주요 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| 벡터(Vector) | "화살표" | 순서가 있는 숫자 목록. AI에서는 고차원 공간의 한 점. |
| 행렬(Matrix) | "숫자로 된 표" | 선형 변환. 한 공간의 벡터를 다른 공간으로 매핑. |
| 행렬 곱셈(Matrix multiply) | "그냥 숫자 곱하기" | 첫 번째 행렬의 모든 행과 두 번째 행렬의 모든 열 사이의 내적. 순서가 중요. |
| 전치(Transpose) | "뒤집기" | 행과 열을 교환. m x n 행렬을 n x m 행렬로 변환. 역전파(backpropagation)에서 중요. |
| 행렬식(Determinant) | "행렬에서 나온 어떤 숫자" | 행렬이 면적(2D) 또는 부피(3D)를 얼마나 확대/축소하는지 측정. 0이면 변환이 차원을 압축. |
| 역행렬(Inverse) | "행렬을 되돌리기" | 변환을 되돌리는 행렬. 행렬식이 0이 아닐 때만 존재. |
| 단위 행렬(Identity matrix) | "지루한 행렬" | 1을 곱하는 것과 같은 행렬. 잔차 연결(ResNets)에서 사용. |
| 브로드캐스팅(Broadcasting) | "마법 같은 모양 맞추기" | 작은 배열을 누락된 차원을 따라 반복하여 큰 배열과 일치시킴. |
| 요소별 연산(Element-wise) | "일반 곱셈" | 일치하는 위치끼리 곱하기. 두 배열은 같은 형태여야 함(또는 브로드캐스팅 가능해야 함).

## 추가 학습 자료

- [3Blue1Brown: 선형 대수의 본질](https://www.3blue1brown.com/topics/linear-algebra) - 여기서 다루는 모든 연산에 대한 시각적 직관
- [NumPy 브로드캐스팅 문서](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy가 따르는 정확한 규칙
- [Stanford CS229 선형 대수 복습](http://cs229.stanford.edu/section/cs229-linalg.pdf) - ML 특화 선형 대수 요약 자료