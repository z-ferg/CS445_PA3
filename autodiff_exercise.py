import scalarflow as sf

graph = sf.Graph()

with graph:
    x = sf.Variable(7.0, name='x')
    y = sf.Variable(-7.0)

    print(graph.run(x))
    print(x.value)
    print(x.derivative)