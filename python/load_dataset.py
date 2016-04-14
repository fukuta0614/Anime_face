import subprocess
import six.moves.cPickle as pickle
import os
from PIL import Image
import numpy as np

class AnimeFaceDataset:
    def __init__(self):
        self.original_dataset_path = u"../data/animeface-character-dataset/thumb/"
        self.data = None
        self.target = None
        self.index2name = {}
        self.n_types_target = -1
        self.dataset_path = u'../data/animeface.dat'
        self.image_size = 32

    def load_data_target(self):
        if os.path.exists(self.dataset_path):
            self.load_dataset()
        else:
            if not os.path.exists(self.original_dataset_path):
                self.fetch_animeface_dataset()

            dir_list = self.get_dir_list()
            self.target = []
            target_name = []
            self.data = []
            for dir_name in dir_list:
                file_list = os.listdir(self.original_dataset_path+dir_name)
                for file_name in file_list:
                    root, ext = os.path.splitext(file_name)
                    if ext == u'.png':
                        abs_name = self.original_dataset_path+dir_name+'/'+file_name
                        # read class id i.e., target
                        image = Image.open(abs_name)
                        image = np.array(image.resize((64,64)), np.float32).transpose(2,0,1)
                        image /= 255.
                        if image.shape != (3,64,64):
                            continue
                        class_id = self.get_class_id(abs_name)
                        self.target.append(class_id)
                        target_name.append(str(dir_name))
                        self.data.append(image)

            for i in xrange(len(self.target)):
                self.index2name[self.target[i]] = target_name[i]

        # print(self.data)
        self.data = np.array(self.data, np.float32)
        self.target = np.array(self.target, np.int32)

        self.dump_dataset()

    def fetch_animeface_dataset(self):
        subprocess.call('wget http://www.nurs.or.jp/\~nagadomi/animeface-character-dataset/data/animeface-character-dataset.zip', shell=True)
        subprocess.call('unzip animeface-character-dataset.zip', shell=True)
        subprocess.call('rm animeface-character-dataset.zip', shell=True)

    def get_dir_list(self):
        tmp = os.listdir(self.original_dataset_path)
        if tmp is None:
            print('download error')
            raise
        return sorted([x for x in tmp if os.path.isdir(self.original_dataset_path+x)])

    def get_class_id(self, fname):
        dir_list = self.get_dir_list()
        dir_name = filter(lambda x: x in fname, dir_list)
        return dir_list.index(dir_name[0])

    def get_n_types_target(self):
        if self.target is None:
            self.load_data_target()

        if self.n_types_target is not -1:
            return self.n_types_target

        tmp = {}
        for target in self.target:
            tmp[target] = 0
        return len(tmp)

    def dump_dataset(self):
        pickle.dump((self.data,self.target,self.index2name), open(self.dataset_path, 'wb'), -1)

    def load_dataset(self):
        self.data, self.target, self.index2name = pickle.load(open(self.dataset_path, 'rb'))

if __name__ == '__main__':
    dataset = AnimeFaceDataset()
    dataset.load_data_target()
