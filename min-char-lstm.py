"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
data = open('data/input.txt', 'r').read() # should be simple plain text file
#data = data.split() # word

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

# model parameters
Wxh = np.random.randn(hidden_size*4, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size*4, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size*4, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lossFun(inputs, targets, hprev, cprev):
    xs, hs, cs, is_, fs, os, gs, ys, ps= {}, {}, {}, {}, {}, {}, {}, {}, {}
    hs[-1] = np.copy(hprev) # t=0일때 t-1 시점의 hidden state가 필요하므로
    cs[-1] = np.copy(cprev)
    loss = 0
    H = hidden_size
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        tmp = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh  # hidden state
        is_[t] = sigmoid(tmp[:H])
        fs[t] = sigmoid(tmp[H:2 * H])
        os[t] = sigmoid(tmp[2 * H: 3 * H])
        gs[t] = np.tanh(tmp[3 * H:])
        cs[t] = fs[t] * cs[t-1] + is_[t] * gs[t]
        hs[t] = os[t] * np.tanh(cs[t])

    # compute loss
    for i in range(len(targets)):
        idx = len(inputs) - len(targets) + i
        ys[idx] = np.dot(Why, hs[idx]) + by  # unnormalized log probabilities for next chars
        ps[idx] = np.exp(ys[idx]) / np.sum(np.exp(ys[idx]))  # probabilities for next chars
        loss += -np.log(ps[idx][targets[i], 0])  # softmax (cross-entropy loss)

    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext, dcnext = np.zeros_like(hs[0]), np.zeros_like(cs[0])
    n = 1
    a = len(targets) - 1
    for t in reversed(range(len(inputs))):
        if n > len(targets):
            continue
        dy = np.copy(ps[t])
        dy[targets[a]] -= 1  # backprop into y
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext  # backprop into h
        dc = dcnext + (1 - np.tanh(cs[t]) * np.tanh(cs[t])) * dh * os[t]  # backprop through tanh nonlinearity
        dcnext = dc * fs[t]
        di = dc * gs[t]
        df = dc * cs[t-1]
        do = dh * np.tanh(cs[t])
        dg = dc * is_[t]
        ddi = (1 - is_[t]) * is_[t] * di
        ddf = (1 - fs[t]) * fs[t] * df
        ddo = (1 - os[t]) * os[t] * do
        ddg = (1 - np.tanh(gs[t]) * np.tanh(gs[t])) * dg
        da = np.hstack((ddi.ravel(),ddf.ravel(),ddo.ravel(),ddg.ravel()))
        dWxh += np.dot(da[:,np.newaxis],xs[t].T)
        dWhh += np.dot(da[:,np.newaxis],hs[t-1].T)
        dbh += da[:, np.newaxis]
        dhnext = np.dot(Whh.T, da[:, np.newaxis])
        n += 1
        a -= 1
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1], cs[len(inputs) - 1]

def sample(h, c, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    H = hidden_size

    for t in range(n):
        tmp = np.dot(Wxh, x) + np.dot(Whh, h) + bh  # hidden state
        i = sigmoid(tmp[:H])
        f = sigmoid(tmp[H:2 * H])
        o = sigmoid(tmp[2 * H: 3 * H])
        g = np.tanh(tmp[3 * H:])
        c = f * c + i * g
        h = o * np.tanh(c)

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
        cprev = np.zeros((hidden_size,1)) # reset RNN memory

        p = 0 # go from start of data
        count = 1
        sample_ix = sample(hprev, cprev, n, 1500)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n {} \n----'.format(txt, ))
        n += 1


    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if count % 100 == 0 or count == total: print('({}/{}), loss: {}'.format(count, total, smooth_loss)) # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    count += 1


'''
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1)) # reset RNN memory
        cprev = np.zeros((hidden_size,1)) # reset RNN memory

        p = 0 # go from start of data

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # sample from the model now and then
    if n % 100 == 0:
        sample_ix = sample(hprev, cprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n {} \n----'.format(txt, ))

    # forward seq_length characters through the net and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev, cprev = lossFun(inputs, targets, hprev, cprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0: print('iter {}, loss: {}'.format(n, smooth_loss)) # print progress

    # perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    n += 1 # iteration counter
'''