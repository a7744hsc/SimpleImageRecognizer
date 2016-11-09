import pickle
from scipy import misc
import time
import numpy as np
import os

PIXEL_DEPTH = 256
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


def pick_folder(dirname):
    if not os.path.isdir(dirname):
        raise Exception('Wrong dir path')
    image_files = os.listdir(dirname)
    dataset = []
    num_image = 0
    for file in image_files:
        full_path = os.path.join(dirname, file)
        image = misc.imread(full_path)
        if image.ndim == 3:  # colour image
            fix_size_image = misc.imresize(image, (IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)
            regularized_image = (fix_size_image - PIXEL_DEPTH / 2) / PIXEL_DEPTH
            dataset.append(regularized_image)
            num_image += 1
        else:
            print('skip black-white image:'+file)
    data_np = np.array(dataset,dtype=np.float32)

    for curser in range(0, len(data_np), 600):  # due to python Issue24658
        try:
            with open(dirname+str(curser)+'.pickle', 'wb') as f:
                pickle.dump(data_np[curser:curser+600], f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', dirname, ':', e)

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def unpick(filename):
    f = open(filename,'rb')
    datas = pickle.load(f)
    return datas

def load_train_set():
    os.chdir('dataset')
    cat0 = unpick('cats0.pickle')
    labels0 = np.zeros(len(cat0), int)
    cat600 = unpick('cats600.pickle')
    labels1 = np.zeros(len(cat600),int)
    cat1200 = unpick('cats1200.pickle')
    labels2 = np.zeros(len(cat1200), int)
    cat1800 = unpick('cats1800.pickle')
    labels3 = np.zeros(len(cat1800), int)

    dogs0 = unpick('dogs0.pickle')
    labels4 = np.ones(len(dogs0), int)
    dogs600 = unpick('dogs600.pickle')
    labels5 = np.ones(len(dogs600), int)
    dogs1200 = unpick('dogs1200.pickle')
    labels6 = np.ones(len(dogs1200), int)

    return np.concatenate((cat0,cat600,cat1200,cat1800,dogs0,dogs600,dogs1200), axis=0), \
           np.concatenate((labels0,labels1,labels2,labels3,labels4,labels5,labels6))


if __name__ == '__main__':
    pick_folder('dogs')
    pick_folder('cats')
    # before = time.clock()
    # dataset,label = load_train_set()
    # after = time.clock()
    # print('da:',dataset.shape)
    # print('label:',label)
    # print('duration:',after-before)

