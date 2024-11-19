import scalarflow as sf

with sf.Graph() as graph:
    x = sf.Variable(0.0, name="x")
    y = sf.Variable(2.0, name="y")

    ex = sf.Exp(x)
    x_squared = sf.Pow(x, 2)
    y_squared = sf.Pow(y, 2)

    xy_sum = sf.Add(x_squared, y_squared)
    
    xy_cube = sf.Pow(xy_sum, 3)

    func = sf.Multiply(xy_cube, ex)

graph.gen_dot("sample.dot")