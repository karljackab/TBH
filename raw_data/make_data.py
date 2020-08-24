import os
from keras.preprocessing import image
from keras.applications import Xception
from keras.applications.xception import preprocess_input
from keras.layers import Input
import numpy as np
from tqdm import tqdm
import csv
import scipy.io as sio
import yaml
import sys
import preprocess

class Dataset():
    def __init__(self, mode=None, data_pth='cifar10', class_len=10):
        ## get the images path
        if mode is not None:
            store_data_pth = os.path.join(data_pth, mode)
        else:
            store_data_pth = data_pth

        ## get class idx mapping list
        self.class_idx_mapping = dict()
        if data_pth == 'cifar10':
            with open(data_pth+'_extract/class_mapping.csv', 'r') as f:
                rows = csv.reader(f)
                for row in rows:
                    self.class_idx_mapping[row[0]] = int(row[1])

        self.class_len = class_len
        self.imgs_pth = []

        ## process cifar10 dataset
        ## for single label task and the images is put in its folder respectively
        if data_pth == 'cifar10':
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
        ## process MSCOCO dataset
        elif data_pth == 'coco':
            if mode == 'test':
                mode = 'val'
            label_pth = f'coco/{mode}_label'

            with open(label_pth, 'r') as f:
                for row in f.readlines():
                    row = row.strip()
                    img_name, label_list = row.split(':')
                    label_list = list(map(lambda x: int(x),label_list.split(',')))

                    img_mode = img_name.split('_')[1]
                    if img_mode == 'train2014':
                        img_pth = 'coco/train2014'
                    elif img_mode == 'val2014':
                        img_pth = 'coco/val2014'
                    file_pth = os.path.join(img_pth, img_name)
                    self.imgs_pth.append((file_pth, label_list))
        ## process NUS-WIDE dataset
        elif data_pth == 'nuswide':
            if mode == 'test':
                mode = 'val'
            label_pth = f'nuswide/{mode}_label'

            with open(label_pth, 'r') as f:
                for row in f.readlines():
                    row = row.strip()
                    file_pth, label_list = row.split(':')
                    label_list = list(map(lambda x: int(x),label_list.split(',')))

                    self.imgs_pth.append((file_pth, label_list))

        self.data_len = len(self.imgs_pth)

    def __getitem__(self, index):
        img_pth, img_label = self.imgs_pth[index]
        sparse_label = np.zeros(self.class_len)
        sparse_label[img_label] = 1
        sparse_label = np.expand_dims(sparse_label, axis=0)

        img = image.load_img(img_pth, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        return img_data, sparse_label, img_pth


def extract_data(extractor, output_pth, feature_size, mode_list=None, data_pth='cifar10', class_len=10, save_labels=True):
    """
    extract image feature
    """
    if data_pth != 'cifar10':
        mode_list.append('database')
    if not os.path.exists(output_pth):
        os.makedirs(output_pth)

    ## build dataset list
    if mode_list is None:
        dataset_list = [Dataset(None, data_pth, class_len)]
    else:
        dataset_list = [Dataset(mode, data_pth, class_len) for mode in mode_list]

    cur_work_dir = os.getcwd()
    for dataset_idx, dataset in enumerate(dataset_list):
        data_size = dataset.data_len

        tot_feature, tot_labels = np.zeros((data_size, feature_size)), np.zeros((data_size, class_len))
        tot_img_pths = []
        cur_size = 0
        data = np.zeros((BS, 224, 224, 3))
        for data_idx in tqdm(range(data_size)):
            img_data, sparse_label, img_pth = dataset[data_idx]

            ## pack current label to total labels variable
            tot_labels[data_idx, :] = sparse_label

            ## pack current image path to total images paths
            tot_img_pths.append(img_pth)

            ## pack input data
            data[cur_size, :, :, :] = img_data

            ## to see it is packed as full batch or not
            cur_size += 1
            if cur_size < BS and data_idx != data_size-1:
                continue

            ## extract feature and reset input data to zero for next batch
            features = extractor.predict(data)
            data = np.zeros((BS, 224, 224, 3))

            ## pack current feature to total features
            tot_feature[data_idx-cur_size+1:data_idx+1, :] = features[0:cur_size]

            cur_size = 0

        ## get current mode ('train' or 'test')
        if mode_list is not None:
            cur_mode = mode_list[dataset_idx]
        else:
            cur_mode = 'all'

        ## write image path list to output file
        with open(os.path.join(output_pth, f'{data_pth}_{cur_mode}_pth_list.csv'), 'w') as pth_fw:
            for img_pth in tot_img_pths:
                pth_fw.write(f'{os.path.join(cur_work_dir, img_pth)}\n')
        
        ## write image feature and labels to output file
        if save_labels:
            result = {
                'data': tot_feature,
                'label': tot_labels
            }
        else:
            result = {'data': tot_feature}
        sio.savemat(os.path.join(output_pth, f'{data_pth}_{cur_mode}.mat'), result)

if __name__ == '__main__':
    ## check and parse config file name
    if len(sys.argv) < 2:
        print('ERROR: config name must be specified')
        print('Usage: python3 make_data.py $config_name')
        exit(1)

    with open(f'{sys.argv[1]}.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('============================')
    print('========== Config ==========')
    for key in config:
        print(f'{key}:\t{config[key]}')
    print('============================')

    ###################
    ## Hyperparameter
    mode_list = ['train', 'test']
    data_pth = config['task']                   ## extract data path
    class_len = config['class_len']             ## class number of this task 
    output_pth = '../data'   ## output feature path
    save_labels = config['save_labels']         ## whether to save images label
    feature_size = config['feature_size']       ## feature length
    BS = config['BS']                           ## batch size
    ###################

    ## do the preprocessing based on different task
    if data_pth == 'cifar10':
        if not os.path.exists(data_pth+'_extract/class_mapping.csv'):
            preprocess.write_class_mapping(mode_list, data_pth)
    elif data_pth == 'coco':
        preprocess.coco_preprocess()
    elif data_pth == 'nuswide':
        if not os.path.exists(f'nuswide/database_label') or \
                not os.path.exists(f'nuswide/train_label') or \
                not os.path.exists(f'nuswide/val_label'):
            preprocess.nuswide_preprocess()

    # extractor = Xception(weights='imagenet', include_top=False, pooling='avg')
    extractor = Xception(weights=None, include_top=False, pooling='avg')
    print(extractor.summary())
    if config['weight_pth'] is not None:
        print(f'load weight {config["weight_pth"]}')
        extractor.load_weights(config['weight_pth'], by_name=True)

    ## extract data
    extract_data(extractor, output_pth, feature_size, mode_list, data_pth, class_len, save_labels)