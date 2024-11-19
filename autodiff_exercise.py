import scalarflow as sf

graph = sf.Graph()

with graph:
    x = sf.Variable(0)
    exp = sf.Exp(x)

    print(graph.run(exp, compute_derivatives=True))
    print(x.value)
    print(x.derivative)