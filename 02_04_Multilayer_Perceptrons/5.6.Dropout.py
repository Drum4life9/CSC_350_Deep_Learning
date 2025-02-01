import torch
from torch import nn
from d2l import torch as d2l
from matplotlib import pyplot as plt
plt.ion()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)

class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(),
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(),
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))

if __name__ == '__main__':
    # X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
    # print('dropout_p = 0:', dropout_layer(X, 0))
    # print('dropout_p = 0.5:', dropout_layer(X, 0.5))
    # print('dropout_p = 1:', dropout_layer(X, 1))

    hparams = {'num_outputs': 10, 'num_hiddens_1': 256, 'num_hiddens_2': 256,
               'dropout_1': 0.5, 'dropout_2': 0.5, 'lr': 0.1}
    model = DropoutMLPScratch(**hparams)
    data = d2l.FashionMNIST(batch_size=256)
    trainer = d2l.Trainer(max_epochs=20)
    trainer.fit(model, data)
    plt.show()

    # model = DropoutMLP(**hparams)
    # trainer.fit(model, data)







"""
Problem 1:
    In some ways, traditional complexity measures are based on systems that are
    inherently based on linear and non-linear functions, such as notation of O(n^2),
    O(n*log(n)), and other measures. With neural networks, apart from the constants
    of weights, we have a lot more freedom to express our functions as combinations of
    linear functions, but sometimes they could even include compositions (as one output
    gets fed into the input of the next function). Especially as we start to get different
    numbers of nodes in each hidden layer, complexities get very tricky and very non-linear
    quickly (even in a way that the 'conventional' standards of complexity are unable to express).

Problem 2:
    Early stopping may be considered a regularization technique as it stops parameters from
    updating earlier than when they "would have" stopped updating. It also helps to prevent
    overfitting, as we stop updating parameters when we reach a state that's "good enough" for
    some definition of this phrase. By stopping early, we work to prevent the model from fitting
    to too much noise detail, which is typically incredibly effective in helping to prevent
    overfitting.

Problem 3:
    If the model is already "mostly" correct, further training would only cause an increase
    in sensitivity to noise in the data, which tends to then lead to overfitting the training
    data set too much. When the overall training error seems to be decreasing a very little bit
    (less than some suggested epsilon from the reading) then it might be a good time to consider
    stopping early to save time and prevent the model from fitting to too much noise.

Problem 4:

"""