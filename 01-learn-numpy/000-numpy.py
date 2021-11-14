import numpy as np

x = np.array([1, 2, 3])
print(x)
print(type(x))

y = np.array([4, 5, 6])

# 사칙연산을 하면 각각의 원소끼리 연산을 수행한다.
print(x + y)
print(x - y)
print(x * y)
print(x / y)

# 벡터와 스칼라의 곱으로 취급
print(x * 2)

# 각각의 원소가 2보다 큰가 판별
print(x > 2)

# 이렇게 하면 2보다 큰 원소만 가져옴
print(x[x > 2])
