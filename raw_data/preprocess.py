import os
import csv
import random

def write_class_mapping(mode_list=['train', 'test'], data_pth='cifar10'):
    """
    create class and idx mapping file
    the target path should conatin folders for which indicated class name
    """
    if not os.path.exists(data_pth+'_extract'):
        os.mkdir(data_pth+'_extract')
    class_idx_mapping = dict()

    ## record every class in train and test folder
    if mode_list is not None:
        for mode in mode_list:
            cur_pth = os.path.join(data_pth, mode)
            for class_name in os.listdir(cur_pth):
                file_pth = os.path.join(cur_pth, class_name)
                if os.path.isdir(file_pth) and class_name not in class_idx_mapping:
                    class_idx_mapping[class_name] = len(class_idx_mapping)

    ## store it to output file
    with open(data_pth+'_extract/class_mapping.csv', 'w') as f:
        writer = csv.writer(f)
        class_idx_mapping_list = [[key, class_idx_mapping[key]] for key in class_idx_mapping]
        writer.writerows(class_idx_mapping_list)

def nuswide_preprocess():
    """
    read the image name and their labels
    only pick out top 21 most frequent categories
    """
    file_pth = 'nuswide/AllLabels81.txt'
    image_list_pth = 'nuswide/Imagelist.txt'

    ## store them to files
    if not os.path.exists('nuswide'):
        os.makedirs('nuswide')

    labels = []
    cls_cnt = [[idx, 0] for idx in range(81)]   ## count the images number of each class
    database = []
    with open(file_pth, 'r') as f:
        for row in f.readlines():
            row = row.strip().split(' ')
            assert len(row) == 81
            label = [idx for idx, lab in enumerate(row) if lab == '1']
            database.append(label)
            for item in label:
                cls_cnt[item][1] += 1

    cls_cnt = sorted(cls_cnt, key=lambda x: x[1], reverse=True)[:21]

    valid_class = [data[0] for data in cls_cnt] ## get final 21 class index
    del cls_cnt

    valid_img = []
    with open(image_list_pth, 'r') as f:
        for idx, row in enumerate(f.readlines()):
            fail = True
            for sub_cls in database[idx]:
                if sub_cls in valid_class:  ## if this image has valid class, store it
                    fail = False
                    break
            if fail:
                continue

            row = "/".join(row.strip().split('\\')[2:])
            valid_img.append((row, database[idx]))

    random.shuffle(valid_img)

    ## get validation set
    val_set = dict() ## key: class, value: (pth, whole_class)
    for cur_cls in valid_class:
        val_set[cur_cls] = []
        remove_idx_list = []
        cnt = 0
        for idx, item in enumerate(valid_img):
            if cur_cls in item[1]:
                cnt += 1
                val_set[cur_cls].append(item)
                remove_idx_list.append(idx)
            if cnt >= 100:
                break
        remove_idx_list.reverse()
        for i in remove_idx_list:   ## delete the stored images in reverse order (to avoid index mismatch)
            del valid_img[i]
        assert len(val_set[cur_cls]) == 100
    assert len(val_set) == 21

    with open(f'nuswide/database_label', 'w') as fw:
        for data in valid_img:
            img_name, labels = data
            labels = [str(label) for label in labels]
            img_name = os.path.join('nuswide', img_name)
            fw.write(f"{img_name}:{','.join(labels)}\n")

    ## get training set
    train_set = dict() ## key: class, value: (pth, whole_class)
    for cur_cls in valid_class:
        train_set[cur_cls] = []
        remove_idx_list = []
        cnt = 0
        for idx, item in enumerate(valid_img):
            if cur_cls in item[1]:
                cnt += 1
                train_set[cur_cls].append(item)
                remove_idx_list.append(idx)
            if cnt >= 500:
                break
        remove_idx_list.reverse()
        for i in remove_idx_list:
            del valid_img[i]
        assert len(train_set[cur_cls]) == 500
    assert len(train_set) == 21

    with open(f'nuswide/train_label', 'w') as fw:
        for key in train_set:
            for data in train_set[key]:
                img_name, labels = data
                labels = [str(label) for label in labels]
                img_name = os.path.join('nuswide', img_name)
                fw.write(f"{img_name}:{','.join(labels)}\n")

    with open(f'nuswide/val_label', 'w') as fw:
        for key in val_set:
            for data in val_set[key]:
                img_name, labels = data
                labels = [str(label) for label in labels]
                img_name = os.path.join('nuswide', img_name)
                fw.write(f"{img_name}:{','.join(labels)}\n")

def coco_preprocess():
    """
    store the images path and their labels based on previous sampled files from other paper
    """
    for mode in ['train', 'val', 'database']:
        file_pth = f'coco/{mode}.txt'

        with open(file_pth, 'r') as f, open(f'coco/{mode}_label', 'w') as fw:
            for row in f.readlines():
                row = row.strip().split(' ')
                img_name, labels = row[0], row[1:]
                img_name = img_name.split('/')[-1]
                assert len(labels) == 80
                labels = [str(idx) for idx, lab in enumerate(labels) if lab == '1']
                fw.write(f"{img_name}:{','.join(labels)}\n")