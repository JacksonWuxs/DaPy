def simple_linear_reg(x, y):
    x_bar, y_bar = sum(x) / len(x), sum(y) / len(y)
    l_xx = sum(map(lambda x: (x - x_bar) ** 2, x))
    l_xy = sum(map(lambda x, y: (x - x_bar) * (y - y_bar), x, y))
    slope = l_xy / float(l_xx)
    return slope, y_bar - slope * x_bar
