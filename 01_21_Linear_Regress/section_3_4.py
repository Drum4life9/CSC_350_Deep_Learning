import torch
import matplotlib.pyplot as plt
from d2l import torch as d2l
plt.ion()


class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""

    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)


@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return torch.matmul(X, self.w) + self.b


@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return l.mean()


class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""

    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)


@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch


@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1


model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
plt.show()

with torch.no_grad():
    print(f'error in estimating w: {data.w - model.w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')

# ------------------------------------------------------------------------------------------------------------------
# Question 1:
#   If we make all the weights 0, the algorithm should still work. We would be trying to find the gradient
#   where all the weights are 0, and this hopefully isn't 0. If it isn't, then all the weights should get updated
#   every iteration of the algorithm. If all our weights were initialized with var = 1,000 it would certainly
#   probably take A LOT longer to converge, but it should still work.

# Question 2:
#   Well, I changed epochs on line 68 to = 10 (from 3) and the error for the B coefficient dropped from around .22 to
#   .0005. The two w coefficient errors are also negligible, with e-06 and e-04 respectively on my first run. I got
#   very similar results the next few runs as well
