
from torch.utils.data import Dataset
import pickle
import torch
import numpy as np

class MyDataset(Dataset):
    def __init__(self, filename='data-science-2023-hw3-few-shot-learning/release/train.pkl', filename2='data-science-2023-hw3-few-shot-learning/release/validation.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        with open(filename2, 'rb') as f:
            data2 = pickle.load(f, encoding='bytes')
        self.images = list(data['images'])  # 38400x3x84x84
        self.labels = list(data['labels'])  # 38400
        self.images.extend(list(data2['images']))
        labels2 = data2['labels']+64
        self.labels.extend(list(labels2))

        self.images = torch.tensor(np.array(self.images))  # 38400x3x84x84
        self.labels = torch.tensor(np.array(self.labels))  # 38400

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]

        label = self.labels[index]
        return image, label

class TestDataset(Dataset):
    def __init__(self, filename='data-science-2023-hw3-few-shot-learning/release/test.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        self.sup_images = torch.tensor(data['sup_images'])  # 600x25x3x84x84
        self.sup_labels = torch.tensor(data['sup_labels'])  # 600x25
        # self.val_images = torch.zeros((600, 5, 3, 84, 84))
        # self.val_labels = torch.zeros((600, 5))
        # self.sup_images_new = torch.zeros((600, 20, 3, 84, 84))
        # self.sup_labels_new = torch.zeros((600, 20))
        # for i in range(600):
        #     for c in range(5):
        #         self.val_images[i][c] = self.sup_images[i][self.sup_labels[i] == c][0]
        #         self.sup_images_new[i][4*c:4*c +
        #                             4] = self.sup_images[i][self.sup_labels[i] == c][1:]
        #         self.val_labels[i][c] = self.sup_labels[i][self.sup_labels[i] == c][0]
        #         self.sup_labels_new[i][4*c:4*c +
        #                             4] = self.sup_labels[i][self.sup_labels[i] == c][1:]

        self.qry_images = torch.tensor(data['qry_images'])  # 600x25x3x84x84

    def __len__(self):
        return len(self.sup_labels)

    def __getitem__(self, index):
        sup_image_nospil = self.sup_images[index]

        sup_label_nospil = self.sup_labels[index]

        # sup_image_20 = self.sup_images_new[index]

        # sup_label = self.sup_labels_new[index]

        qry_image = self.qry_images[index]

        # val_image = self.val_images[index]

        # val_label = self.val_labels[index]

        return sup_image_nospil, sup_label_nospil,qry_image
