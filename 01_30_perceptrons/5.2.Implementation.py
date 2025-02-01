import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
plt.ion()

d2l.use_svg_display()


class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))


def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = X.reshape((-1, self.num_inputs))
    H = relu(torch.matmul(X, self.W1) + self.b1)
    return torch.matmul(H, self.W2) + self.b2

if __name__ == '__main__':
    model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
    data = d2l.FashionMNIST(batch_size=256)
    trainer = d2l.Trainer(max_epochs=2)
    trainer.fit(model, data)
    plt.show()


class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))

#model = MLP(num_outputs=10, num_hiddens=32, lr=0.1)
#trainer.fit(model, data)


# I'll be coming to you at some point, my programs are not running so answering
#   these questions is hard :(
#
#   Problem 1) This makes the inner hidden layer wider. I ran this using
#   colab with only 5 epochs instead of 10. 32 num_hiddens produced a validation
#   accuracy of around .8, while 256 produced one higher than that, even on
#   epoch 3. For time's sake I'm going to use 32 num_hiddens

#   Problem 2) I used 32 hidden nodes per layer with 2 hidden layers.
#   After training for 5 epochs, the validation accuracy was still
#   Hovering around .8, but the validation loss was higher than it was
#   with only 1 layer. Not sure how that works, but on the 5th epoch the loss
#   did decrease by a lot.

#   Problem 3) Epochs @ 10 and LR at 1. So far based on the initial few epochs,
#   The model is doing significantly worse at training loss and val loss, but
#   accuracy is almost close to 0! Higher numbers create some funky things
#   going on. LR = .05, the losses and such slowed down descending, but
#   it will more slowly approach the true value.