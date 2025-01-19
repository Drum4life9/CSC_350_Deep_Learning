import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)

@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)

@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)

model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2, num_train=10000)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)

@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)
w, b = model.get_w_b()

print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
print(f'error in estimating b: {data.b - b}')

# --------------------------------------------------------------------------------------------------------------
# Question 3:
#   a) Trying to get plots to work. Hopefully will get it attached to the assignment
#   b) The hint is appropriate because it takes an exponentially bigger amount (resulting in a logarithmic scale
#       to normalize it) in order for more data to become more beneficial towards the overall error rate. Also,
#       adding data tends to significantly decrease overfitting since the model gets more practice at being able
#       to generalize results to new & unseen data

# Question 4:
#   K-fold cross validation is typically very expensive to compute since it requires each K < n fold to be used as
#   a training set K - 1 times, and a validation set 1 time. We essentially have to end up training K different models,
#   and depending on K and n, each training time could be quite expansively large