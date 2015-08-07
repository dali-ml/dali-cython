from test_dali import Mat, random, MatOps, Graph

num_examples = 100
example_size = 3
iterations   = 150
lr           = 0.01

X = random.uniform(
    0.0,
    1.0 / example_size,
    size=(num_examples, example_size)
)
ones = Mat.ones((X.shape[1], 1))
Y = X.dot(ones)

X = MatOps.consider_constant(X)
Y = MatOps.consider_constant(Y)

W = random.uniform(-1.0, 1.0, (example_size, 1))
print(repr(W))
for i in range(iterations):
    predY = X.dot(W)
    error = ((predY - Y) ** 2).sum()
    print(repr(error))
    # line below can be replaced by simply error.grad()
    error.dw += 1
    Graph.backward()
    # there are much nicer solvers in Dali,
    # but here we write out gradient descent
    # explicitly
    W.w -= W.dw * lr
    W.dw = 0
print(repr(W))
