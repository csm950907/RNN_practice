"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD license
python 3에서 실행 가능하도록 수정, 한글 해설 추가
from : https://gist.github.com/MinjeJeon/8f50693f0a986419ab2dda35753acb1f
original : https://gist.github.com/karpathy/d4dee566867f8291f086
"""
import numpy as np

# 데이터를 불러오고, 글자-벡터 간 상호 변환 매핑 준비
data = open('data/input.txt', 'r').read() # 텍스트 파일 로드
chars = list(set(data)) # 텍스트 파일에서 고유한 문자 추출
data_size, vocab_size = len(data), len(chars)
print('데이터는 {}개의 글자로 되어 있고, {}개의 고유한 문자가 있습니다.'.format(data_size, vocab_size))
print(repr(''.join(sorted(str(x) for x in chars)))) # 추출된 고유한 글자들을 알파벳 순서대로 출력

# 고유한 글자들(a,b,c,d...)을 숫자(1,2,3,4...)에 매핑하는 사전과, 반대 기능을 수행하는 사전을 만듦
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# 하이퍼파라미터 설정
hidden_size = 100 # hidden state의 뉴런 갯수
seq_length = 25 # 학습시킬 때 한번에 불러올 글자 수이자 RNN을 펼쳤을 때의 단계
learning_rate = 1e-1 # 학습속도, 가중치를 조정할 때 이동할 간격

# 모델 파라미터 초기화(가중치는 작은 수의 랜덤한 값, bias는 0으로 초기화)
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden (100,25)
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden (100,100)
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output (25,100)
bh = np.zeros((hidden_size, 1)) # hidden bias (100,1)
by = np.zeros((vocab_size, 1)) # output bias (25,1)

def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    inputs, targets는 모두 숫자 인덱스의 리스트이다.
    hprev는 H(hidden_size)x1의 array, 이전 학습에서 반환한 마지막 hidden state임
    forward pass(손실값 계산), backward pass(그래디언트 계산)를 모두 수행한 후
    손실값, 각각의 가중치에 대한 그래디언트, 그리고 다음 반복 때 사용할 마지막 hidden state를 반환함.
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass(손실값 계산)
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # 1-of-k(one-hot) 형태로 변환. 모든 값이 0인 array 준비
        xs[t][inputs[t]] = 1 # 해당하는 글자에만 값을 1로 설정 - [0, ..., 0, 1, 0, ..., 0]
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state 업데이트
        ys[t] = np.dot(Why, hs[t]) + by # 다음 글자가 어떤 글자가 나올지에 가능성을 표시한 array(정규화되지 않음)
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # softmax로 각 글자의 등장 가능성을 확률로 표시
        loss += -np.log(ps[t][targets[t],0]) # cross-entropy를 이용하여 정답과 비교하여 손실값 판정
    # backward pass(그래디언트 계산)
    # 변수 초기화
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))): #forward pass의 과정을 반대로 진행(t=24부터 시작)
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # y의 그래디언트 계산, softmax 함수의 그래디언트 계산
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext # loss에서 사용된 h와 h를 업데이트한 계산의 그래디언트 값을 더함
        dhraw = (1 - hs[t] * hs[t]) * dh # tanh 역전파
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam) # 그래디언트 발산 방지
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    모델에서 지정된 글자 수(n) 만큼의 글자(숫자의 리스트)를 출력
    h 는 hidden state, seed_ix는 주어진 첫번째 글자
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        # forward pass 수행
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))

        # 샘플링. 임의성을 부여하기 위해 argmax대신 array p에서 주어진 확률에 의해 하나의 문자를 선택
        ix = np.random.choice(range(vocab_size), p=p.ravel())

        # 다음 글자 추론을 위해 샘플링 된 글자를 다음 입력으로 사용
        x = np.zeros((vocab_size, 1))
        x[ix] = 1

        # 결과값 리스트에 추가
        ixes.append(ix)
    return ixes


n, p = 0, 0 #  반복 회수(n) 및 입력 데이터(p) 위치 초기화

# Adagrad 알고리즘에 사용되는 메모리 변수 초기화
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length # 학습이 이루어지기 전의 손실값
while True:
    # 입력데이터 준비, 텍스트의 맨 앞쪽부터 seq_length만큼씩 데이터를 준비
    # 데이터를 모두 사용하면 입력 데이터의 맨 처음으로 이동
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1)) # RNN 메모리 초기화
        p = 0 # 입력 데이터의 맨 처음으로 이동

    # 입력(p~p+24번째 글자), 목표(p+1~p+25번째 글자) 데이터를 준비
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # 학습을 100번 반복할 때마다 학습 결과를 출력
    if n % 1000 == 0:
        sample_ix = sample(hprev, inputs[0], 5000) #지금까지 학습한 RNN을 이용하여 숫자의 리스트를 출력
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

    # 손실함수에서 손실값과 그래디언트를 함께 계산
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 1000 == 0: print('iter %d, loss: %f' % (n, smooth_loss)) # 반복횟수, 손실 출력

    # Adagrad 방식으로 파라미터 업데이트
    for param, dparam, mem in zip([Wxh,  Whh,  Why,  bh,  by],   # 가중치
                                  [dWxh, dWhh, dWhy, dbh, dby],  # 그래디언트
                                  [mWxh, mWhh, mWhy, mbh, mby]): # 메모리
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # 실제 파라메터 업데이트

    p += seq_length # 데이터 포인터를 seq_length만큼 우측으로 이동
    n += 1 # 반복횟수 카운터
