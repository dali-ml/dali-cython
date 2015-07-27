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
    error.grad()
    Graph.backward()
    MatOps.sgd_update(W, lr)
    W.clear_grad()
print(repr(W))
