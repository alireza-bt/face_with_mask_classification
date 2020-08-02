import random
import os
from pathlib import Path
from shutil import copyfile

PATH = 'D:\\How To\\Data Science\\Projects\\data\\Face Mask Classification'


def rename_dataset_items():
    # rename items if needed
    class_names = []
    for c in os.listdir(PATH):
        class_names.append(c)

    for c in class_names:
        counter = 1000
        try:
            for _, _, files in os.walk(os.path.join(PATH, c)):
                for item in files:
                    if os.path.isfile(os.path.join(PATH, c, str(item))):
                        item_path = os.path.join(PATH, c, str(item))
                        name = item.split('.')
                        name[0] = '_'.join([c, 'img', str(counter)])
                        new_name = '.'.join(name)
                        counter += 1
                        os.rename(item_path, os.path.join(PATH, c, new_name))
        except Exception as e:
            print(e)
        if counter == 0:
            print('There is no item in class \'%s\'' % c)
        else:
            print('changed %s items in class \'%s\'' % (str(counter), c))


def split_data():
    all_data = []
    class_names = []
    for c in os.listdir(PATH):
        class_names.append(c)

    for c in class_names:
        try:
            for _, _, files in os.walk(os.path.join(PATH, c)):
                for item in files:
                    if os.path.isfile(os.path.join(PATH, c, str(item))):
                        all_data.append(item)
        except Exception as e:
            print(e)

    print('all items in different classes: %s' % str(len(all_data)))

    all_data.sort()  # make sure that the filenames have a fixed order before shuffling
    random.seed(230)
    random.shuffle(all_data)  # shuffles the ordering of filenames (deterministic given the chosen seed)

    split_1 = int(0.8 * len(all_data))
    split_2 = int(0.9 * len(all_data))
    train_filenames = all_data[:split_1]
    dev_filenames = all_data[split_1:split_2]
    test_filenames = all_data[split_2:]

    try:
        Path(os.path.join(PATH, 'train\\with')).mkdir(mode=0o777, parents=True, exist_ok=True)
        Path(os.path.join(PATH, 'train\\without')).mkdir(mode=0o777, parents=True, exist_ok=True)
        Path(os.path.join(PATH, 'test\\with')).mkdir(mode=0o777, parents=True, exist_ok=True)
        Path(os.path.join(PATH, 'test\\without')).mkdir(mode=0o777, parents=True, exist_ok=True)
        Path(os.path.join(PATH, 'dev\\with')).mkdir(mode=0o777, parents=True, exist_ok=True)
        Path(os.path.join(PATH, 'dev\\without')).mkdir(mode=0o777, parents=True, exist_ok=True)
        for item in train_filenames:
            class_name = item.split('_')[0]
            copyfile(os.path.join(PATH, class_name, str(item)), os.path.join(PATH, 'train', class_name, str(item)))
        print('copied %s items to train folder' % (str(len(train_filenames))))

        for item in test_filenames:
            class_name = item.split('_')[0]
            copyfile(os.path.join(PATH, class_name, str(item)), os.path.join(PATH, 'test', class_name, str(item)))
        print('copied %s items to test folder' % (str(len(test_filenames))))

        for item in dev_filenames:
            class_name = item.split('_')[0]
            copyfile(os.path.join(PATH, class_name, str(item)), os.path.join(PATH, 'dev', class_name, str(item)))
        print('copied %s items to dev folder' % (str(len(dev_filenames))))
    except Exception as e:
        print(e)


if __name__ == "__main__":
    rename_dataset_items()
    split_data()