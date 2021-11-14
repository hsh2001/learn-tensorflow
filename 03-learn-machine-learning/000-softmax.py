import numpy as np


# 소프트맥스 함수 구현.
# 근데 안전하지 않다. 이유는 아래서!!
def unsafe_softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


# 적당한 값의 요소들은 잘 계산된다!
a = np.array([1, 12, 3, 4])
print(unsafe_softmax(a))


# 만약 요소의 값이 대략 1000 보다 커지면
# 계산이 제대로 안된다.
# exp(1000)의 값은 너무 커서
# 제대로 계산을 못한다.

a = np.array([10000, 12222, 11133, 13299])
print(unsafe_softmax(a))


# 소프트맥스 함수의 안전한 구현
def softmax(a):
    # a 리스트에 스칼라값 c를 빼도 기존의 a와 연산 결과는 같다!
    # https://t1.daumcdn.net/cfile/tistory/99EFEB335D8C14752F
    max_item = np.max(a)
    new_a = max_item - a
    exp_a = np.exp(new_a)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


a = np.array([1, 12, 3, 33])
print(softmax(a))

# 잘 작동한다!!
a = np.array([1010, 1000, 990])
print(softmax(a))

# 소프트 맥스 함수의 특징은 출력의 합이 1이란 점이다.
a = np.array([0.3, 0.2, 0.9])
sum = np.sum(softmax(a))
print(sum)
