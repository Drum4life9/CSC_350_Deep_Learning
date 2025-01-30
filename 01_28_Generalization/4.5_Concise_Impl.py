import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

print('Hello, I\'m not dead')

class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)

@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
    Y = Y.reshape((-1,))
    return F.cross_entropy(
        Y_hat, Y, reduction='mean' if averaged else 'none')

data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=5)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)

# Problem 1: 4.5.3
#   As the number of epochs increases for training, the validation accuracy
#   may decrease for a number of reasons, the primary one being potential
#   overfitting of the data. As we train more and more on the training data,
#   the model will likely more closely resemble the training dataset, so when
#   it comes to validation data, it may perform slightly worse (hard to tell)
#   if the model has overfitted to the training data, meaning it would not
#   generalize well. We could fix this through a number of different techniques,
#   but it may be better to train less in order for a model to generalize more.
#   We could also employ other strategies (dropout?) to force models to train
#   on different features of the data set to hopefully make all the nodes slightly
#   more well-rounded in their effect on the output.

# Problem 2: 4.5.4
#   As we increase the learning rate, the backpropagation step can sometimes cause the
#   the updates that are made to be over or undershot. Since we multiply the loss by
#   learning rate, this likely will cause our updates to swing very drastically around the
#   true value of the minimum of the gradient for loss. Typically, smaller losses (between .01
#   to .1 depending on the problem at hand) tend to work better, as the descent is enough to
#   make consistent progress toward the actual value, but not too much in order to overshoot
#   the actual value of the weight for the true f(x). This is a pretty standard starting range
#   for many types of problems, but there is obviously a lot more flexibility you can have
#   with models

# Problem 3:
#   a) If we have a model trained on identifying cats, dogs, cows, and pigs and we feed it a
#   chicken, the model will have no care in the world that it's a chicken, it will use its
#   pre-trained skills to identify it as either a cat, dog, pig, or cow. The overall
#   probabilities likely will be a lot lower, but it will still give a concise answer
#   based on the outputs it was trained to give
#
#   b) In a perfect world, we'd probably want the model to realize that all the probabilities
#   are close for the output (or some way to identify that it's not one of the trained animals)
#   and to suggest that you added an animal that it was not trained on. We probably could just
#   adjust the handling of the softmax output to check the closeness of the probability values
#   to see if there is anything that is unusual for a model trained on those 4 specific animals.
#   One such implementation could be some sort of probabilistic comparison to see how close the
#   output probabilities are, and make a decision whether to raise a flag about a new animal being
#   added. 