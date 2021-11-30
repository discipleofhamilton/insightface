import argparse

import cv2
import numpy as np
import torch
import os
import math
import time

from backbones import get_model

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

def get_distance(emb1, emb2, mode = 0):
    if mode == 0:
        # Euclidian distance
        diff = np.subtract(emb1, emb2)
        dist = np.sqrt(np.sum(np.square(diff)))
        # diff = np.subtract(emb1, emb2)
        # dist = np.sum(np.square(diff), 1)
    elif mode == 1:
        # Euclidian L2 distance
        diff = np.subtract(emb1, emb2)
        dist = np.sqrt(np.sum(np.square(diff)))
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        norm = np.sqrt(np.sum(np.square(emb1))) * np.sqrt(np.sum(np.square(emb2)))
        dist = dist / norm
        # diff = np.subtract(emb1, emb2)
        # dist = np.sum(np.square(diff), 1)
    elif mode==2:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(emb1, emb2))
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        similarity = max(-1, min(1, (dot / norm)))
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % mode 
        
    return dist

def stable_sigmoid(x, base:int = 1):

    if x >= 0:
        z = math.exp(-x)
        sig = base / (base + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (base + z)
        return sig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Testing')
    parser.add_argument('image_files', type=str, nargs='+', help='Images to compare')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--threshold', type=float, default=0.4)
    argv = parser.parse_args()

    embeddings, _ = get_features(argv.weight, argv.network, argv.image_files)
    names         = [os.path.basename(img_path).split(".")[0] for img_path in argv.image_files]

    # euclidean 4
    distances     = [[get_distance(e1, e2, mode=1) for e2 in embeddings] for e1 in embeddings]
    # confidences   = 1 - np.array(distances)
    confidences   = (4 - np.square(np.array(distances))) / 4

    print("\neuclidean 4")
    print("names", names)
    print("distances: \n", np.around(np.array(distances), 4))
    print("confidences: \n", np.around(confidences, 4))

    # euclidean 2
    distances     = [[get_distance(e1, e2, mode=1) for e2 in embeddings] for e1 in embeddings]
    # confidences   = 1 - np.array(distances)
    confidences   = (2 - np.array(distances)) / 2

    print("\neuclidean 2")
    print("names", names)
    print("distances: \n", np.around(np.array(distances), 4))
    print("confidences: \n", np.around(confidences, 4))

    # cosine
    distances     = [[get_distance(e1, e2, mode=2) for e2 in embeddings] for e1 in embeddings]
    confidences   = 1 - np.array(distances)
    fusion_conf   = np.empty(confidences.shape)
    sigmoid_conf  = np.empty(confidences.shape)

    w, _ = confidences.shape

    for i in range(w):
        for j in range(w):
            if distances[i][j] >= argv.threshold:
                fusion_conf[i,j] = confidences[i,j]**2
            else:
                fusion_conf[i,j] = np.sqrt(confidences[i,j])

    for i in range(w):
        for j in range(w):
            sigmoid_conf[i,j] = stable_sigmoid(confidences[i,j]*10, 0.5)

    print("\ncosine")
    print("names", names)
    print("fusion_conf: \n",  np.around(fusion_conf, 4))
    print("sigmoid_conf: \n",  np.around(sigmoid_conf, 4))
    print("distances: \n", np.around(np.array(distances), 4))
    print("confidences: \n", np.around(confidences, 4))