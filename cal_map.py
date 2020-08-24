import os
from util.eval_tools import eval_cls_map
import matplotlib.pyplot as plt
import scipy.io as sio
import util.others as other_utils

def get_all_record(pth):
    ## it will get all .mat files, only return the unique timestep and sort the list
    record_list = os.listdir(pth)
    timestep_list = set()
    for item in record_list:
        file_type = item.split('.')[-1]
        if file_type != 'mat':
            continue
        timestep = item.split('.')[0].split('_')[-1]
        try:
            timestep_list.add(int(timestep))
        except:
            continue
    return sorted(list(timestep_list))

def calculate_map(code_folder, at_top_k, title, result_folder):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    ## get all the generated code name
    timestep_list = get_all_record(code_folder)

    ## get MAP and plot MAP curve
    MAP_list = []
    for timestep in timestep_list:
        if task == 'cifar10':
            db_prefix = 'train'
        else:
            db_prefix = 'database'
        ## load the code to "db_record" and "test_record"
        db_record, test_record = \
            f'{db_prefix}_{task}_{code_length}_{timestep}.mat', f'test_{task}_{code_length}_{timestep}.mat'
        db_record, test_record = \
            sio.loadmat(os.path.join(code_folder, db_record)), \
            sio.loadmat(os.path.join(code_folder, test_record))

        test_code, test_label, db_code, db_label = \
            test_record['code'], test_record['label'], db_record['code'], db_record['label']
        print(f'Run {timestep}')
        print(f'test_code shape: {test_code.shape}, test_label shape: {test_label.shape}')
        print(f'db_code shape: {db_code.shape}, db_label shape: {db_label.shape}')

        ## calculate Mean Average Precision
        MAP = eval_cls_map(test_code, db_code, test_label, db_label, at=at_top_k)

        ## write the MAP to output file
        print(f'{timestep}: {MAP}')
        with open(os.path.join(result_folder, f'{title}_record'), 'a') as f:
            f.write(f'{timestep}: {MAP}\n')

        ## append MAP to MAP_list, for following visualization
        MAP_list.append(MAP)
        print('========================')

    ## visualize MAP curve, store it to png file
    plt.xlabel('time')
    plt.ylabel('MAP')
    plt.title(title)
    plt.plot(timestep_list, MAP_list)
    plt.savefig(os.path.join(result_folder, f'{title}.png'))

if __name__ == '__main__':
    config = other_utils.read_config()
    #####################################
    task = config['task']
    code_length = config['code_length']
    at_top_k = config['at_top_k']
    code_folder = f'data/code/{task}_{code_length}bit/'
    title = f'{task}_{code_length}bit'
    result_folder = 'data/result'
    #####################################

    calculate_map(code_folder, at_top_k, title, result_folder)