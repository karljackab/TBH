import os
import keras
from keras.preprocessing import image
import numpy as np
import csv
import preprocess
import random
from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))

def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

class Dataset(keras.utils.Sequence):
    def __init__(self, mode=None, data_pth='cifar10', class_len=10, BS=32, img_size=224, preprocess_method=None):
        super().__init__()
        ## get the images path
        if mode is not None:
            store_data_pth = os.path.join(data_pth, mode)
        else:
            store_data_pth = data_pth

        ## get class idx mapping list
        self.class_idx_mapping = dict()
        self.mode = mode
        self.class_len = class_len
        self.BS = BS
        self.img_size = img_size
        self.imgs_pth = []
        self.preprocess_method = preprocess_method
        self.datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        ## process cifar10 dataset
        ## for single label task and the images is put in its folder respectively
        if data_pth == 'cifar10':
            if not os.path.exists('cifar10_extract/class_mapping.csv'):
                preprocess.write_class_mapping(['train', 'test'], 'cifar10')
            with open('cifar10_extract/class_mapping.csv', 'r') as f:
                rows = csv.reader(f)
                for row in rows:
                    self.class_idx_mapping[row[0]] = int(row[1])

            folder_list = [(store_data_pth, None)]
            iter_idx = 0

            while iter_idx < len(folder_list):
                cur_pth, cur_label_name = folder_list[iter_idx]
                cur_label = self.class_idx_mapping[cur_label_name] if cur_label_name is not None else -1

                ## get all the images and their labels, the label would be it's folder name
                for file_name in os.listdir(cur_pth):
                    file_pth = os.path.join(cur_pth, file_name)
                    if os.path.isdir(file_pth):
                        folder_list.append((file_pth, file_name))
                    else:
                        self.imgs_pth.append((file_pth, cur_label))
                iter_idx += 1
        # ## process MSCOCO dataset
        # elif data_pth == 'coco':
        #     if mode == 'test':
        #         mode = 'val'
        #     label_pth = f'coco/{mode}_label'

        #     with open(label_pth, 'r') as f:
        #         for row in f.readlines():
        #             row = row.strip()
        #             img_name, label_list = row.split(':')
        #             label_list = list(map(lambda x: int(x),label_list.split(',')))

        #             img_mode = img_name.split('_')[1]
        #             if img_mode == 'train2014':
        #                 img_pth = 'coco/train2014'
        #             elif img_mode == 'val2014':
        #                 img_pth = 'coco/val2014'
        #             file_pth = os.path.join(img_pth, img_name)
        #             self.imgs_pth.append((file_pth, label_list))
        ## process NUS-WIDE dataset
        elif data_pth == 'nuswide':
            if not os.path.exists(f'nuswide/database_label') or \
                    not os.path.exists(f'nuswide/train_label') or \
                    not os.path.exists(f'nuswide/val_label'):
                preprocess.nuswide_preprocess()

            if mode == 'test':
                mode = 'val'
            # elif mode == 'train':
            #     mode = 'database'
            label_pth = f'nuswide/{mode}_label'

            with open(label_pth, 'r') as f:
                for row in f.readlines():
                    row = row.strip()
                    file_pth, label_list = row.split(':')
                    label_list = list(map(lambda x: int(x),label_list.split(',')))

                    self.imgs_pth.append((file_pth, label_list))
        else:
            raise Exception("TASK NOT EXIST!")

        self.batch_len = (len(self.imgs_pth) // BS)-1
        self.idx_list = list(range(len(self.imgs_pth)))
        random.shuffle(self.idx_list)

    def __len__(self):
        return self.batch_len

    def __data_gen(self, s_idx):
        x = np.empty((self.BS, self.img_size, self.img_size, 3))
        y = np.empty((self.BS, self.class_len))

        sub_idx = 0
        for idx in self.idx_list[s_idx:s_idx+self.BS]:
            img_pth, img_label = self.imgs_pth[idx]
            sparse_label = np.zeros(self.class_len)
            sparse_label[img_label] = 1
            sparse_label = np.expand_dims(sparse_label, axis=0)

            img = image.load_img(img_pth, target_size=(self.img_size, self.img_size))
            img_data = image.img_to_array(img)
            x[sub_idx] = img_data
            y[sub_idx] = sparse_label
            sub_idx += 1
        assert sub_idx == self.BS
        
        if self.mode == 'train':
            x = self.datagen.flow(x, shuffle=False).next()

        return x, y

    def __getitem__(self, index):
        x, y = self.__data_gen(index*self.BS)
        if self.preprocess_method is not None:
            x = self.preprocess_method(x)
        return x, y