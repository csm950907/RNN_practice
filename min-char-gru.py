import numpy as np

data = open('./data/input.txt', 'r').read()

chars = list(set(data)) # Get number of unique characters
data_size, vocab_size = len(data), len(chars)
print('data has {} characters, {} unique.'.format(data_size, vocab_size))

# char_to_ix : dict{'character':'index'}
# ix_to_char : dict{'index':'character'}
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

Wxh = np.random.randn(hidden_size * 3, vocab_size)      # Wxh, Wxh_z, Wxh_r
Whh = np.random.randn(hidden_size * 3, hidden_size)     # Whh, Whh_z, Whh_r
Why = np.random.randn(vocab_size, hidden_size)
bh = np.zeros((hidden_size * 3, 1))              # bh, bh_z, bh_r
by = np.zeros((vocab_size, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lossFun(inputs, targets, hprev):
    xs, hs, zs, rs, rs_, hs_, ys, ps = {}, {}, {}, {}, {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    H = hidden_size

    # forward propagation
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        W = np.dot(Wxh, xs[t])                         # hidden state, shape(300, 1)
        U = np.dot(Whh, hs[t - 1])                     # hidden state, shape(300, 100)
        zs[t] = sigmoid(W[:H] + U[:H] + bh[:H])
        rs[t] = sigmoid(W[H:H * 2] + U[H:H * 2] + bh[H:H * 2])
        rs_[t] = rs[t] * U[H * 2:]
        hs_[t] = np.tanh(W[H * 2:] + rs_[t])
        hs[t] = zs[t] * hs[t-1] + (1 - zs[t]) * hs_[t]

    # compute loss
    for i in range(len(targets)):
        idx = len(inputs) - len(targets) + i
        ys[idx] = np.dot(Why, hs[idx]) + by  # unnormalized log probabilities for next chars
        ps[idx] = np.exp(ys[idx]) / np.sum(np.exp(ys[idx]))  # probabilities for next chars
        loss += -np.log(ps[idx][targets[i], 0])  # softmax (cross-entropy loss)

    # backward propagation
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        dh_ = dh * (1 - zs[t])
        dz = dh * (np.diag(hs[t-1].ravel()) - np.diag(hs_[t].ravel()))
        dr_ = dh_ * (1 - np.tanh(W[H * 2:] + rs_[t]) * np.tanh(W[H * 2:] + rs_[t]))
        dr = dr_ * np.diag(U[H * 2:].ravel())
        dinput_r = np.dot(dr, rs[t] * (1 - rs[t]))
        dinput_z = np.dot(dz, zs[t] * (1 - zs[t]))
        dWxh_x = dh_*(1 - np.tanh(W[H * 2:] + rs_[t]) * np.tanh(W[H * 2:] + rs_[t]))
        dWhh_h = np.dot(np.diag(rs[t].ravel()), dr_)
        dall_x = np.hstack((dinput_z.ravel(), dinput_r.ravel(), dWxh_x.ravel()))
        dall_h = np.hstack((dinput_z.ravel(), dinput_r.ravel(), dWhh_h.ravel()))
        dWxh += np.dot(dall_x[:, np.newaxis], xs[t].T)
        dWhh += np.dot(dall_h[:, np.newaxis], hs[t-1].T)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    H = hidden_size

    for t in range(n):
        W = np.dot(Wxh, x)                         # hidden state, shape(300, 1)
        U = np.dot(Whh, h)                     # hidden state, shape(300, 100)
        z = sigmoid(W[:H] + U[:H] + bh[:H])
        r = sigmoid(W[H:H * 2] + U[H:H * 2] + bh[H:H * 2])
        r_ = r * U[H * 2:]
        h_ = np.tanh(W[H * 2:] + r_)
        h = z * h + (1 - z) * h_

        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes

n, p, count = 0, 0, 1
total = int(data_size / seq_length)
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

while n < 10:
    if p+seq_length+1 >= len(data) or n == 0:
        print('================\n iteration {} \n================'.format(n + 1))
        hprev = np.zeros((hidden_size,1)) # reset RNN memory

        p = 0 # go from start of data
        count = 1
        sample_ix = sample(hprev, n, 1500)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n {} \n----'.format(txt, ))
        n += 1


    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if count % 1000 == 0 or count == total: print('({}/{}), loss: {}'.format(count, total, smooth_loss)) # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    count += 1
