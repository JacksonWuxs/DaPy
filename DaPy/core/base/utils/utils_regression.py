def simple_linear_reg(x, y):
    x_bar, y_bar = sum(x) / len(x), sum(y) / len(y)
    l_xx = sum(map(lambda x: (x - x_bar) ** 2, x), 0.0)
    l_xy = sum(map(lambda x, y: (x - x_bar) * (y - y_bar), x, y), 0.0)
    if l_xx == 0:
        return 0, y_bar
    slope = l_xy / l_xx
    return slope, y_bar - slope * x_bar
