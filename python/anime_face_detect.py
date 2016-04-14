import time
import six.moves.cPickle as pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from chainer import cuda, optimizers


from load_dataset import AnimeFaceDataset
from imagenet import ImageNet


class CNN:
    def __init__(self, data, target, n_outputs, gpu=-1, index2name=None):

        self.model = ImageNet(n_outputs)
        self.model_name = 'cnn_model'

        if gpu >= 0:
            self.model.to_gpu()

        self.gpu = gpu

        self.x_train,\
        self.x_test,\
        self.y_train,\
        self.y_test = train_test_split(data, target, test_size=0.1)

        self.n_train = len(self.y_train)
        self.n_test = len(self.y_test)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        self.train_loss = []
        self.test_loss = []

        self.index2name = index2name

    def predict(self, x_data, gpu=-1):
        index = self.model.predict(x_data, gpu)
        if self.index2name is None:
            return index
        else:
            return self.index2name[index]

    def train_and_test(self, n_epoch=100, batchsize=100):

        epoch = 1
        best_accuracy = 0
        while epoch <= n_epoch:
            print 'epoch', epoch

            perm = np.random.permutation(self.n_train)
            sum_train_accuracy = 0
            sum_train_loss = 0
            for i in xrange(0, self.n_train, batchsize):
                x_batch = self.x_train[perm[i:i+batchsize]]
                y_batch = self.y_train[perm[i:i+batchsize]]

                real_batchsize = len(x_batch)

                self.optimizer.zero_grads()
                loss, acc = self.model.forward(x_batch, y_batch, train=True, gpu=self.gpu)
                loss.backward()
                self.optimizer.update()
                # print(type(loss))
                sum_train_loss += float(loss.data.get()) * real_batchsize
                sum_train_accuracy += float(acc.data.get()) * real_batchsize

            print 'train mean loss={}, accuracy={}'.format(sum_train_loss/self.n_train, sum_train_accuracy/self.n_train)
            self.train_loss.append(sum_train_accuracy/self.n_train)

            # evaluation
            sum_test_accuracy = 0
            sum_test_loss = 0
            for i in xrange(0, self.n_test, batchsize):
                x_batch = self.x_test[i:i+batchsize]
                y_batch = self.y_test[i:i+batchsize]

                real_batchsize = len(x_batch)

                loss, acc = self.model.forward(x_batch, y_batch, train=False, gpu=self.gpu)

                sum_test_loss += float(cuda.to_cpu(loss.data)) * real_batchsize
                sum_test_accuracy += float(cuda.to_cpu(acc.data)) * real_batchsize

            print 'test mean loss={}, accuracy={}'.format(sum_test_loss/self.n_test, sum_test_accuracy/self.n_test)
            self.test_loss.append(sum_test_accuracy/self.n_test)
            epoch += 1
        plt.plot(self.test_loss)
        plt.plot(self.train_loss)
        print('...save graph')
        plt.savefig('result.png')

    def dump_model(self):
        self.model.to_cpu()
        pickle.dump(self.model, open(self.model_name, 'wb'),-1)

    def load_model(self):
        self.model = pickle.load(open(self.model_name,'rb'))
        if self.gpu >= 0:
            self.model.to_gpu()
        self.optimizer.setup(self.model)


if __name__ == '__main__':
    print '...loading AnimeFace dataset'
    dataset = AnimeFaceDataset()
    dataset.load_data_target()
    data = dataset.data
    target = dataset.target
    index2name = dataset.index2name
    n_outputs = dataset.get_n_types_target()

    print '...building model'
    cnn = CNN(data=data,
              target=target,
              gpu=0,
              n_outputs=n_outputs,
              index2name=index2name
              )

    cnn.train_and_test(n_epoch=100)
    cnn.dump_model()