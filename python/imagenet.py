import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable
import numpy as np


class ImageNet(chainer.Chain):
    def __init__(self, n_outputs):
        super(ImageNet, self).__init__(
                conv1=L.Convolution2D(3, 32, 5),
                conv2=L.Convolution2D(32, 32, 7),
                l3=L.Linear(2048, 2048),
                l4=L.Linear(2048, n_outputs)
        )

    def forward(self, x_data, y_data, train=True, gpu=-1):

        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)
            y_data = cuda.to_gpu(y_data)

        x, t = Variable(x_data), Variable(y_data)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)
        h = F.dropout(F.relu(self.l3(h)), train=train)
        y = self.l4(h)
        return F.softmax_cross_entropy(y, t), F.accuracy(y,t)

    def predict(self, x_data, gpu=-1):

        if gpu >= 0:
            x_data = cuda.to_gpu(x_data)

        x = Variable(x_data)
        h = F.max_pooling_2d(F.relu(self.conv1(x)), ksize=2, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), ksize=3, stride=3)
        h = F.dropout(F.relu(self.l3(h)), train=False)
        y = self.l4(h)
        # print(y.data,np.argmax(y.data))
        return np.argmax(y.data)
