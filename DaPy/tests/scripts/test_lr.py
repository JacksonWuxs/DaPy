import DaPy as dp
from DaPy.methods import LinearRegression as dp_lr

data = dp.read('advertising.csv')
lr_dp = dp_lr('numpy')
lr_dp.fit(data['TV':'newspaper'], data['sales'])
lr_dp.report.show()
