import torch
import matplotlib.pyplot as plt
plt.ion()
from d2l import torch as d2l

X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdims=True), X.sum(1, keepdims=True))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here

X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(1))

class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = X.reshape((-1, self.W.shape[0]))
    return softmax(torch.matmul(X, self.W) + self.b)


y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0, 1], y])

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()

print(cross_entropy(y_hat, y))

@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)


data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
plt.show()

X, y = next(iter(data.val_dataloader()))
preds = model(X).argmax(axis=1)
print(preds.shape)

wrong = preds.type(y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)

# Question 4.4.1:
#   a) Nope, e^100 is likely going to be way too big for any standard calculator system to compute
#   b) Again, now e^-100 will be so close to 0 (around 44 0s after the decimal!) that even if this was
#       the largest value, there's no way a computer would be able to efficiently handle this
#   c) We could implement some sort of scaling based on the largest value, so almost like we could
#       normalize the data before running it through the softmax (which normalizes again), but we could
#       center our mean around 0 before running through softmax


# Question 4.4.3
#   Not exactly, there may be cases where you want to return the most likely labels, in which case
#       we'd need to develop some sort of normalization for the outputs (maybe against another softmax!)
#       to show the relative probabilities if two (or more) output probabilities are close to each other.
#
#   For medical diagnoses, we would probably certainly want to give multiple options for a particular
#       disease diagnosis, not just "oh this one is .0001 percent more likely, so that's what it has to be"


# Question 4.4.4
#   Large output probability vectors can cause some serious issues, especially as our size increases,
#       the relative outputs of all the probabilities will decrease towards 0 and everything will get
#       shrunken down. This is not helpful for probability predicting, as we want to see clearly distinct
#       probabilistic likelihoods that give us a good indication whether that is a likely output or not.