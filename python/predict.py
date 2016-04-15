import six.moves.cPickle as pickle
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Chainer example: AnimeFace')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--source', '-s', default='', help='path to the source of image')
args = parser.parse_args()

data, target, index2name = pickle.load(open('../data/animeface.dat', 'rb'))
model = pickle.load(open('cnn_model','rb'))

image = Image.open(args.source)
image = np.array(image.resize((64,64)), np.float32).transpose(2,0,1)
image /= 255.
if image.shape != (3,64,64):
    print('the image is too small')
    raise

index = model.forward(image.reshape((1,3,64,64)), args.gpu)

print(index2name[index])