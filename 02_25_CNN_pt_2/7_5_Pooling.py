import torch
from torch import nn
from d2l import torch as d2l


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print(pool2d(X, (2, 2)))
print(pool2d(X, (2, 2), 'avg'))

X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)
pool2d = nn.MaxPool2d(3)
# Pooling has no model parameters, hence it needs no initialization
print(pool2d(X))

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

X = torch.cat((X, X + 1), 1)

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

'''
Question 1, 7.4.2
    1) Computational Cost: Not entirely sure how to answer this exactly.
    We're for sure going to need c0 additions * ci for the number of channels, and
    floor((kh + ph)/sh) * floor((kw + pw)/sw) for total length and width. Maybe
    this is it? I can't tell
    
    2) Not sure exactly what you mean, I assume the number of parameters we're
    going to need to hold. Probably just the same as the computational cost forward
    except the c0 term, since we won't need to hold a parameter for the addition
    of the different channels to the output of the layer.
    
    3) The same? I'm really not sure about this one
    
    4) I feel like this is going to be a lot of derivatives to calculate across
    the channels in the layers
    
Question 2, 7.5.3
    1) I get the general sense, but I can't quite figure out how to get this to
    work. I'm thinking something like: 
    Max(a,b) = (ReLU(a) * ReLU(-b)) + (ReLU(-a) * ReLU(b)) 
    I think this does it!
    
    2) Our convolution kernel I think looks like an w*h matrix filled with 1s, and
    -1 on the diagonal. We need to first take every input, run it through its 
    corresponding place in the matrix, then apply a ReLU to each output individually
    (to get the ReLU(-a) and such), then we'd need to apply some sort of w*h pooling
    operation on top that just adds the elements of that result together. I guess
    that wouldn't necessarily be called a pool, but we need to sum these elements
    
    3) I think for a 2x2 convolution (I'm assuming by this we mean that we're 
    shrinking down by a 2x2 region), we'd need a 2x2 kernel with maybe 2 layers,
    one that has a window of    1, 0, 
                                0, -1
    and one that's              -1, 0
                                0,  1
    that way the appropriate positives and negatives get added up.
    For the 3x3, I think we'd need 3x3 and 3 layers, with 
    1   0   0
    0   1   0
    0   0   -1
    
    1   0   0
    0   -1  0
    0   0   1
    
    -1  0   0
    0   1   0
    0   0   1
    
    I think I have the general idea as to what's going on here, but I dont think
    my structure is 100% correct. Definitely hoping we talk about this in class
    
    Question 3, 7.5.5
    Max-pooling seeks to in essence find the "most important" value of the window
    (for whatever we are defining "most important" to mean in the context of the
    input values, maybe it's brightness of pixel, brighter = more important = 
    higher number), whereas average-pooling seeks to take all the data of the 
    window into consideration and give a result that is more represented by
    all of the data in the window when it comes to the output. 
'''