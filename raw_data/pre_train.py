import keras
from keras.preprocessing import image
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.utils import np_utils
import utils
import numpy as np
import scipy.io as sio
import os
from tqdm import tqdm

cls_mapping = dict()
task = 'nuswide'
class_len = 81
EPOCH = 100
BS = 32
LR = 0.0001
# weight_pth = 'result/cifar10_bk2/7_0.8885.h5'
weight_pth = None

class SelfCallback(keras.callbacks.Callback):
    def __init__(self, task):
        super().__init__()
        self.cur_best_val_acc = 0.
        if not os.path.exists(f'result/{task}'):
            os.makedirs(f'result/{task}')

    def on_epoch_end(self, epoch, logs={}):
        print(f'{epoch}, {logs}')
        if self.cur_best_val_acc < logs['f1']:
            self.cur_best_val_acc = logs['f1']
            self.model.save_weights(f'result/{task}/{epoch}_{self.cur_best_val_acc:.4f}.h5')


if __name__ == '__main__':
    # model = Xception(weights=None, pooling='avg', classes=class_len)
    # print(model.summary())
    ######################################
    xception_model = Xception(weights=None, pooling='avg', classes=class_len)
    xception_model.layers.pop()
    print(xception_model.summary())
    model = Sequential()
    model.add(xception_model)
    model.add(Dense(class_len, activation='sigmoid'))
    print(model.summary())
    ###########################################
    if weight_pth is not None:
        print(f'load weight {weight_pth}')
        model.load_weights(weight_pth)
    # opt = keras.optimizers.SGD(lr=LR, momentum=0.9)
    opt = keras.optimizers.Adam(lr=LR)
    # model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', utils.f1, utils.precision, utils.recall])
    model.compile(loss=utils.create_weighted_binary_crossentropy(0.2, 0.8), optimizer=opt, metrics=[utils.f1, utils.precision, utils.recall])

    train_generator = utils.Dataset(mode='train', data_pth=task, class_len=class_len, BS=BS, img_size=299, preprocess_method=preprocess_input)
    test_generator = utils.Dataset(mode='test', data_pth=task, class_len=class_len, BS=BS, img_size=299, preprocess_method=preprocess_input)
    history = model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        epochs=EPOCH,
                        callbacks=[SelfCallback(task)])

    with open(f'result/{task}/log', 'w') as f:
        f.write('epoch,')
        for key in history.history:
            f.write(f'{key},')
        f.write('\n')
        for i in range(EPOCH):
            f.write(f'{i},')
            for key in history.history:
                f.write(f'{history.history[key][i]},')
            f.write('\n')