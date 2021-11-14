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
    # y 원소중 하나라도 0이면 log(y_k)의 값은 음의 무한대.
    # 참고로 np.log는 밑이 e인 자연로그이다.
    # 보통 밑이 10인 경우 밑을 생략하여 log로 작성하고 밑이 e일 경우 ln으로 작성하는 데,
    # np에서는 log가 상용로그가 아니고 자연로그임에 주의.
    return -np.sum(t * np.log(y + 1e-7))
