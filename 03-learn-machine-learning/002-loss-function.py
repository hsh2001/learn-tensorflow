import numpy as np


# 손실함수, 비용함수 (loss function, cost function)은
# 신경망 성능의 나쁨을 나타내는 지표. 이를 나타내는 함수는 대표적으로
# 1. sum of squares for error (SSE) 오차 제곱의 합
# 2. mean of squares for error (MSE) 오차 제곱의 평균
# 3. root mean of squares for error (RMSE) 오차 제곱의 평균의 제곱근
# 4. cross entropy error (CEE) 교차 엔트로피 오차


# SSE
def sum_squares_error(y, t):
    # 오차의 제곱 ((y - t) ** 2)의 합의 절반.
    return 0.5 * np.sum((y - t) ** 2)


# CEE
def cross_entropy_error(y, t):
    # 주의! y 원소에 0.0001처럼 아주 작은 값을 더해서
    # y원소의 값이 0이 되지 않도록 한다.
    return -np.sum(t * np.log(y + 1e-7))
