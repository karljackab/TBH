import os
import scipy.io as sio
import util.eval_tools as eval_tools
import numpy as np
import time
from tqdm import tqdm
import yaml
import util.others as other_utils

def retrieve(db_code_pth, query_code_pth, db_img_list_pth, query_img_list_pth, \
            retrieve_pth, top_k=10):
    ## create the path if it is not exist
    if not os.path.exists(retrieve_pth):
        os.makedirs(retrieve_pth)

    ## open DB and query images path mapping file, save it as list type
    with open(db_img_list_pth) as f_db, open(query_img_list_pth) as f_q:
        db_imgs_list = list(map(lambda x: x.strip(), f_db.readlines()))
        query_imgs_list = list(map(lambda x: x.strip(), f_q.readlines()))

    ## load the code of DB and query images
    db_code, query_code = sio.loadmat(db_code_pth)['code'], sio.loadmat(query_code_pth)['code']
    
    ## compute the hamming distance between DB and query code, and sort it
    distances = eval_tools.compute_hamming_dist(query_code, db_code)
    dist_argsort = np.argsort(distances)

    query_len = query_code.shape[0]
    for q_idx in tqdm(range(query_len)):
        ## get query images name and type
        temp = query_imgs_list[q_idx].split('/')[-1]
        q_file_name, file_type = temp.split('.')

        ## get current time hashing, to prevent same query image name confict
        hash_time = str(hash(time.process_time()))[-1:-5:-1]
        result_pth = os.path.join(retrieve_pth, f'{q_file_name}_{hash_time}')

        ## make directory and copy query image to it
        os.mkdir(result_pth)
        os.system(f'cp {query_imgs_list[q_idx]} {os.path.join(result_pth, f"query.{file_type}")}')

        ## retrieve top K DB images as results, and copy it to the retrieved folder
        for retrieve_idx in range(top_k):
            db_idx = dist_argsort[q_idx, retrieve_idx]
            db_file_type = db_imgs_list[db_idx].split('.')[1]
            os.system(f'cp {db_imgs_list[db_idx]} {os.path.join(result_pth, f"{retrieve_idx}.{db_file_type}")}')


if __name__ == '__main__':
    config = other_utils.read_config()
    ##################################################
    ## hyperparameter
    task = config['task']
    code_length = config['code_length']
    top_k = config['top_k']                                                          ## retrieve top K images
    ###################
    db_code_pth = f'data/code/{task}_{code_length}bit/train_{task}_{code_length}_eval.mat'   ## database code path
    query_code_pth = f'data/code/{task}_{code_length}bit/test_{task}_{code_length}_eval.mat' ## query code path
    db_img_list_pth = f'data/{task}_train_pth_list.csv'                 ## database image mapping list path
    query_img_list_pth = f'data/{task}_test_pth_list.csv'               ## query image mapping list path
    retrieve_pth = f'retrieve_result/{task}_{code_length}bit'                                    ## retrieve result path
    ##################################################

    retrieve(db_code_pth, query_code_pth, db_img_list_pth, query_img_list_pth, retrieve_pth, top_k)