from model import TBH
from util.dataset import DataHelper, MatDataset, BasicDataset
import tensorflow as tf
from meta import REPO_PATH
import os
import util.others as other_utils

config = other_utils.read_config()
print(f'TF Version: {tf.__version__}')
############
train_file = os.path.join(REPO_PATH, 'data', f'{config["task"]}_train.mat')   ## the training data file
output_folder_name = f'{config["task"]}_{config["code_length"]}bit'
############

model_config = {'batch_size': config['batch_size'], 'code_length': config['code_length'], 
                'input_length': config['input_length'], 'iteration': config['train_iter'], 'LR': config['learning_rate'],
                'output_folder_name': output_folder_name, 'continue_length': config['continue_length']}
train_config = {'batch_size': config['batch_size'], 'code_length': config['code_length'], 
                'file_name': train_file, 'phase': 'train', 'with_ground_label':True}

sess = tf.Session()

## declare model
model = TBH(**model_config)
## prepare data
train_data = MatDataset(**train_config)
data_helper = DataHelper(train_data)

## start training
model.train(sess, data_helper)