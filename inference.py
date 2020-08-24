from model import TBH
from util.dataset import InferenceDataHelper, MatDataset, BasicDataset
import tensorflow as tf
from meta import REPO_PATH
import os
from util.eval_tools import eval_cls_map
import matplotlib.pyplot as plt
import util.others as other_utils

class InferenceWrapper():
    def __init__(self, file_name, task='cifar10', batch_size=400, code_length=32, 
                phase='test', input_length=2048, continue_length=512):
        self.task= task
        self.weight_list = []
        self.phase = phase
        self.batch_size = batch_size
        self.code_length = code_length

        model_config = {
            'batch_size': batch_size,
            'code_length': code_length,
            'input_length': input_length,
            'continue_length': continue_length
        }
        data_config = {
            'batch_size': self.batch_size,
            'code_length': self.code_length,
            'file_name': file_name,
            'phase': 'test',
            'with_ground_label': True
        }

        self.sess = tf.compat.v1.Session()
        self.model = TBH(**model_config)
        self.data_helper = InferenceDataHelper(MatDataset(**data_config))

    def change_data_and_phase(self, new_file_name, new_phase=None):
        new_data_config = {
            'batch_size': self.batch_size,
            'code_length': self.code_length,
            'file_name': new_file_name,
            'phase': 'test',
            'with_ground_label': True
        }
        self.data_helper = InferenceDataHelper(MatDataset(**new_data_config))
        if new_phase is not None:
            self.phase = new_phase

    def extract(self, model_weight, folder='data/code', suffix=None):
        ## extract binary hashing vector to "folder"
        ## if "suffix" is not None, it would append to the tail of code file name

        self.model.extract(self.sess, self.data_helper, restore_file=model_weight,
            folder=folder, task=self.task, suffix=suffix, phase=self.phase)

    def get_all_weight(self, model_pth):
        ## get all of weight file names in "model_pth"
        ## for evaluation of every model weights score only

        model_list = os.listdir(model_pth)
        model_set = set()
        for item in model_list:
            item = item.split('.')
            if len(item) < 2:
                continue
            item = item[0].split('-')[1]
            model_set.add(int(item))
        return sorted(list(model_set))

    def test_all_weight(self, model_pth, folder='data/code'):
        ## evaluate the score of every model weight 

        weight_list = self.get_all_weight(model_pth)
        assert len(weight_list) > 0

        for item in weight_list:
            model_weight = os.path.join(model_pth, f'model-{item}')
            self.extract(model_weight, folder=folder, suffix=item)

        return weight_list

if __name__ == '__main__':
    config = other_utils.read_config()
    #######################################################
    ## Hyperparameter
    task = config['task']
    batch_size = config['batch_size']
    input_length = config['input_length']
    code_length = config['code_length']
    continue_length = config['continue_length']
    phase = 'test'
    ################
    model_pth = os.path.join(REPO_PATH, 'data', 'model', f'{task}_{code_length}bit')
    code_folder= os.path.join(REPO_PATH, 'data', 'code', f'{task}_{code_length}bit')
    file_name = os.path.join(REPO_PATH, 'data', f'{task}_{phase}.mat')
    #######################################################

    wrapper = InferenceWrapper(file_name=file_name, task=task, batch_size=batch_size, code_length=code_length,
                phase=phase, input_length=input_length, continue_length=continue_length)

    if config['usage'] == 1:
        ###### First Usage: generate code from every weight, usually for evaluation MAP
        ## evaluate at every model's weight
        wrapper.test_all_weight(model_pth, code_folder)

        ## change phase and file name
        new_phase = 'database'
        if task == 'cifar10':
            new_phase = 'train'
        new_file_name = os.path.join(REPO_PATH, 'data', f'{task}_{new_phase}.mat')
        wrapper.change_data_and_phase(new_file_name, new_phase)

        ## evaluate again
        wrapper.test_all_weight(model_pth, code_folder)
    elif config['usage'] == 2:
        ###### Second Usage: generate code fomr single weight, usually for inference
        ### extract binary hashing code only, need to specify the weight
        model_weight = config['model_weight']
        wrapper.extract(os.path.join(model_pth, model_weight), code_folder, suffix='eval')

        ## change phase and file name
        new_phase = 'database'
        if task == 'cifar10':
            new_phase = 'train'
        new_file_name = os.path.join(REPO_PATH, 'data', f'{task}_{new_phase}.mat')
        wrapper.change_data_and_phase(new_file_name, new_phase)

        ### extract again
        wrapper.extract(os.path.join(model_pth, model_weight), code_folder, suffix='eval')
    else:
        print('ERROR: "usage" setting error')
        exit(1)