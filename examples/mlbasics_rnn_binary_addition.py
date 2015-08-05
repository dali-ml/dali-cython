import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

import random
from test_dali import Mat, MatOps, Graph, SGD, RNN, Layer

def as_bytes(num, final_size):
    res = []
    for _ in range(final_size):
        res.append(num % 2)
        num //= 2
    return res

def generate_example(num_bits):
    a = random.randint(0, 2**(num_bits - 1) - 1)
    b = random.randint(0, 2**(num_bits - 1) - 1)
    res = a + b
    return (as_bytes(a,  num_bits),
            as_bytes(b,  num_bits),
            as_bytes(res,num_bits))

ITERATIONS_PER_EPOCH = 30
NUM_BITS             = 30
INPUT_SIZE           = 2
OUTPUT_SIZE          = 1
MEMORY_SIZE          = 5
MAX_EPOCHS           = 5000

rnn                  = RNN(INPUT_SIZE, MEMORY_SIZE)
classifier           = Layer(MEMORY_SIZE, OUTPUT_SIZE)
rnn_initial          = Mat(1, MEMORY_SIZE)

solver               = SGD()
solver.step_size     = 0.001
params               = rnn.parameters() + classifier.parameters() + [rnn_initial]

for epoch in range(MAX_EPOCHS):
    for _ in range(ITERATIONS_PER_EPOCH):
        a, b, res = generate_example(NUM_BITS)
        error = Mat.zeros((1,1))
        prev_hidden = rnn_initial
        for bit_idx in range(NUM_BITS):
            input_i = Mat([a[bit_idx], b[bit_idx]], dtype=rnn.dtype)
            print(input_i.dtype)
            prev_hidden = rnn.activate(input_i, prev_hidden).tanh()
            #prev_hidden = (rnn.Wx.dot(input_i) + rnn.Wh.dot(prev_hidden) + rnn.b).tanh()
            output_i    = classifier.activate(prev_hidden).sigmoid()
            # print(repr(output_i))
            error = error + MatOps.binary_cross_entropy(output_i, res[bit_idx])
        error.grad()
        Graph.backward()
    if epoch % 20 == 0:
        print("epoch %d, error = %.3f" % (epoch, error.w[0,0]))
    solver.step(params)
