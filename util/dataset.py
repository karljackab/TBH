import numpy as np
import scipy.io as sio
from util.eval_tools import eval_cls_map
import os

class BasicDataset(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 256)
        self.code_length = kwargs.get('code_length', 32)
        self.phase = kwargs.get('phase', 'train')
        self.data = np.asarray([0])
        self.code = np.zeros((self.set_size, self.code_length))
        self.batch_count = 0
        self.with_ground_label = kwargs.get('with_ground_label', False)
        if self.with_ground_label:
            self.label = np.asarray([0])
        else:
            self.label = None
        self.this_batch = dict()

    @property
    def set_size(self):
        return self.data.shape[0]

    @property
    def batch_num(self):
        return self.set_size // self.batch_size

    def _shuffle(self):
        index = np.arange(self.set_size)
        np.random.shuffle(index)
        self.data = self.data[index, ...]
        if self.with_ground_label:
            self.label = self.label[index, ...]
        self.code = self.code[index, ...]

    def next_batch(self):
        pass

    def update(self, code: np.ndarray):
        batch_start = self.this_batch.get('batch_start')
        batch_end = batch_start + self.batch_size
        self.code[batch_start:batch_end, ...] = code
        self.this_batch['batch_code'] = code


class MatDataset(BasicDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_name = kwargs.get('file_name')
        self._load_data()
        self.code = np.zeros((self.set_size, self.code_length))

    def _load_data(self):
        mat_file = sio.loadmat(self.file_name)
        for i in mat_file.keys():
            if i.find('label') >= 0:
                label_key = i
            if i.find('data') >= 0:
                data_key = i
        self.data = np.asarray(mat_file[data_key], dtype=np.float32)

        if self.with_ground_label:
            mat_file[label_key] = mat_file[label_key].squeeze()
            if np.asarray(mat_file[label_key], dtype=np.int32).shape.__len__() == 1:
                sparse_label = np.asarray(mat_file[label_key], dtype=np.int32)
                real_label = np.zeros((self.set_size, np.max(sparse_label) + 1))
                for i in range(self.set_size):
                    real_label[i, sparse_label[i]] = 1
                self.label = real_label
            else:
                self.label = np.asarray(mat_file[label_key], dtype=np.int32)
        else:
            self.label = None
        del mat_file

    def next_batch(self):
        if self.batch_count == 0 and self.phase == 'train':
            self._shuffle()

        batch_start = self.batch_count * self.batch_size
        batch_end = batch_start + self.batch_size

        batch_image = self.data[batch_start:batch_end, ...]
        if self.with_ground_label:
            batch_label = self.label[batch_start:batch_end, ...]
            self.this_batch['batch_label'] = batch_label

        self.batch_count = (self.batch_count + 1) % self.batch_num

        self.this_batch['batch_image'] = batch_image
        self.this_batch['batch_start'] = batch_start
        self.this_batch['batch_end'] = batch_end

        return self.this_batch

class DataHelper(object):
    def __init__(self, training_data: BasicDataset):
        self.training_data = training_data

    def next_batch(self):
        return self.training_data.next_batch()

    def update(self, code: np.ndarray):
        return self.training_data.update(code)

    def hook_train(self):
        if not self.training_data.with_ground_label:
            return None

        q = self.training_data.this_batch['batch_code']
        l = self.training_data.this_batch['batch_label']

        return eval_cls_map(q, q, l, l)

    def save(self, set_name, length, folder='data'):
        if self.training_data.with_ground_label:
            to_save = {'set_code': self.training_data.code,
                        'set_label': self.training_data.label}
        else:
            to_save = {'set_code': self.training_data.code}
        sio.savemat('{}/code/{}_{}.mat'.format(folder, set_name, length), to_save)


class InferenceDataHelper(object):
    def __init__(self, data: BasicDataset):
        self.data = data

    def next_batch(self):
        return self.data.next_batch()

    def update(self, code: np.ndarray):
        return self.data.update(code)

    def save(self, set_name, length, phase='test', folder='data/code', suffix=None):
        if self.data.with_ground_label:
            to_save = {'code': self.data.code,
                       'label': self.data.label}
        else:
            to_save = {'code': self.data.code}

        if not os.path.exists(folder):
            os.makedirs(folder)
        if suffix is not None:
            pth = f'{folder}/{phase}_{set_name}_{length}_{suffix}.mat'
        else:
            pth = f'{folder}/{phase}_{set_name}_{length}.mat'

        sio.savemat(pth, to_save)
