import argparse

import cv2
import numpy as np
import torch
import os
import math
import time

from backbones import get_model

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, np.array(labels_flat)

def get_categories(data_dir):
    path_exp = os.path.expanduser(data_dir)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    # get the label strings
    categories = [name for name in classes if \
       os.path.isdir(os.path.join(path_exp, name))]

    return np.array(categories)

@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    # print("\nnetwork name {}".format(name))
    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(weight, map_location='cuda:0'))
    eval_start = time.time()
    net.eval()
    feat = net(img).numpy()
    return feat
    # print(feat)

@torch.no_grad()
def get_features(weight, name, imgs):

    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(weight, map_location='cuda:0'))
    # net.load_state_dict(torch.load(weight, map_location='cpu:0'))
    net.eval()

    features = list()
    eval_time_list = list()

    for img in imgs:

        if img is None:
            img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        else:
            img = cv2.imread(img)
            img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        # print("\nnetwork name {}".format(name))
    
        eval_start = time.time()
        feat = net(img).numpy()
        eval_end = time.time()

        features.append(feat)
        eval_time_list.append(eval_end - eval_start)

    return features, np.array(eval_time_list)
    # print(feat)

@torch.no_grad()
def get_features_from_db(weight, name, imgs, labels):

    net = get_model(name, fp16=True)
    net.load_state_dict(torch.load(weight, map_location='cuda:0'))
    net.eval()

    features = list()

    for img in imgs:
        
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        # print("\nnetwork name {}".format(name))
    
        feat = net(img).numpy()
        features.append(feat)

    features = np.array(features)
    print(features.shape)

    labels_list, labels_list_indice = np.unique(labels, return_index=True)
    start = -1
    end = -1
    emb_class_array = list()
    for i in range(1, len(labels_list_indice)+1):
        if i == len(labels_list_indice):
            emb_class_array.append(np.mean(features[labels_list_indice[i-1]:, :],axis=0))
        else:
            emb_class_array.append(np.mean(features[labels_list_indice[i-1]:labels_list_indice[i], :],axis=0))
    emb_class_array = np.array(emb_class_array)
    return emb_class_array
    # print(feat)

def get_distance(emb1, emb2, mode = 0):
    if mode == 0:
        # Euclidian distance
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff))
        # diff = np.subtract(emb1, emb2)
        # dist = np.sum(np.square(diff), 1)
    elif mode == 1:
        # Euclidian L2 distance
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff))
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        dist = dist / norm
        # diff = np.subtract(emb1, emb2)
        # dist = np.sum(np.square(diff), 1)
    elif mode==2:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(emb1, emb2))
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % mode 
        
    return dist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Testing')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--database_dir', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.4)
    args = parser.parse_args()

    test_data = get_dataset(args.test_dir)
    test_imgs, test_labels = get_image_paths_and_labels(test_data)
    test_label_names = get_categories(args.test_dir)
    test_label_names = test_label_names[test_labels]

    print("\ntest")
    print("label shape: {}, label name shape: {}, label names: {}\n".format(
        test_labels.shape, test_label_names.shape, test_label_names
    ))

    db_data = get_dataset(args.database_dir)
    db_imgs, db_labels = get_image_paths_and_labels(db_data)
    db_label_names = get_categories(args.database_dir)
    db_label_names = db_label_names[db_labels]

    print("\ndb")
    print("label shape: {}, label name shape: {}, label names: {}\n".format(
        db_labels.shape, db_label_names.shape, db_label_names
    ))

    # db_fs = [inference(args.weight, args.network, img) for img in db_imgs]
    # test_fs = [inference(args.weight, args.network, img) for img in test_imgs]

    # db_fs = get_features_from_db(args.weight, args.network, db_imgs, db_labels)
    db_fs, _ = get_features(args.weight, args.network, db_imgs)
    test_fs, eval_time_list = get_features(args.weight, args.network, test_imgs)

    # db_fs = db_fs.reshape(, 512)
    print(len(db_fs), type(db_fs[0]))
    print(test_fs[0].shape)
    print("\n")

    correct_counter = 0
    outer_list = list()
    err_list = list()
    test_size = len(test_fs)
    for i in range(test_size):

        distances = np.array([get_distance(test_fs[i], db_f, 2) for db_f in db_fs])
        min_dist_ind = np.argmin(distances)

        print(test_label_names[i], distances[min_dist_ind], db_label_names[min_dist_ind])
        # print(test_imgs[i].split("/")[-2], distances[min_dist_ind], db_label_names[min_dist_ind])
        if distances[min_dist_ind] >= args.threshold:
            if test_label_names[i] not in db_label_names:
                correct_counter += 1
            else:
                outer_list.append(test_imgs[i])
            print("\n\timage: {} is not in the database!!!\n".format(test_imgs[i]))
            continue
        else:
            max_key = db_label_names[min_dist_ind]

        if test_label_names[i] == max_key:
            correct_counter += 1
        else:
            err_list.append(test_imgs[i])
        print("\n\tinput category: {}, get embedding time: {:.2f}ms, predict category: {}, conf: {:.2f}\n".format(
            test_label_names[i], eval_time_list[i] * 1000,  db_label_names[min_dist_ind], distances[min_dist_ind]
        ))
    
    accuracy = (correct_counter / test_size) * 100
    print("Accuracy: {:.2f}%, Mean of getting embedding time: {:.2f}ms\n".format(
        accuracy, np.mean(eval_time_list) * 1000
    ))

    print("Wrong predict of category")
    for err in err_list:
        print(err)

    print("wrong right")
    for out in outer_list:
        print(out)