import os
import math
import numpy as np
import pandas as pd
from utils import load_img
from tensorflow.keras.utils import Sequence


class SequenceData(Sequence):

    def __init__(self, model, dir, target_size, batch_size, shuffle=True):
        self.model = model
        self.data_sets = []
        if self.model is 'train':
            with open(os.path.join(dir, '2007_train.txt'), 'r') as f:
                self.data_sets = self.data_sets + f.readlines()
        elif self.model is 'val':
            with open(os.path.join(dir, '2007_val.txt'), 'r') as f:
                self.data_sets = self.data_sets + f.readlines()
        self.image_size = target_size[0:2]
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data_sets))
        self.shuffle = shuffle

    def __len__(self):
        # 计算每一个epoch的迭代次数
        num_imgs = len(self.data_sets)
        return math.ceil(num_imgs / float(self.batch_size))

    def __getitem__(self, idx):
        # 生成batch_size个索引
        batch_indexs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch = [self.data_sets[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch)
        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read(self, dataset):
        # dataset = 'C:\\BaiduNetdiskDownload\\pascalvoc\\VOCdevkit/VOC2007/JPEGImages/000012.jpg 156,97,351,270,6'
        dataset = dataset.strip().split()
        # dataset = ["图片路径","ROI坐标+标签"]
        image_path = dataset[0]
        labels = dataset[1:]

        image, image_h, image_w = load_img(path=image_path, shape=self.image_size)

        label_matrix = np.zeros([7, 7, 25])
        # 7*7的网格，每格做一次20分类（20种目标）+4偏移量回归+1置信度
        for label in labels:
            # 遍历同一张图多个目标
            label = label.split(',')
            # 坐标和分类以逗号分隔
            label = np.array(label, dtype=np.int)
            xmin = label[0]
            ymin = label[1]
            xmax = label[2]
            ymax = label[3]
            cls = label[4]
            x = (xmin + xmax) / 2 / image_w
            y = (ymin + ymax) / 2 / image_h
            # 获取归一化中心坐标
            w = (xmax - xmin) / image_w
            h = (ymax - ymin) / image_h
            # 获取归一化高宽
            loc = [7 * x, 7 * y]
            # 找到该中心坐标映射到网格图后相对所在网格的偏移量
            loc_i = int(loc[1])
            loc_j = int(loc[0])
            y = loc[1] - loc_i
            x = loc[0] - loc_j
            # 把中心坐标映射到网格图的所在网格打上对应的分类标签
            if label_matrix[loc_i, loc_j, 24] == 0:
                # 24位为置信度，如果为1表明已有目标由此方格负责预测，则不做进一步处理
                label_matrix[loc_i, loc_j, cls] = 1
                # 对应标签类置1(0到第19共20类)
                label_matrix[loc_i, loc_j, 20:24] = [x, y, w, h]
                # 第20到第23共四位存储目标框信息（偏移量和高宽）
                label_matrix[loc_i, loc_j, 24] = 1
                # 第24位存储置信度，有目标为1，无目标为0

        return image, label_matrix

    def data_generation(self, batch_datasets):
        images = []
        labels = []

        for dataset in batch_datasets:
            image, label = self.read(dataset)
            images.append(image)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)

        # print(
        #     "Person类占该batch总目标个数比例：",
        #     np.count_nonzero(y[:, :, :, 14]) / np.count_nonzero(y[:, :, :, 0:20])
        # )

        return X, y


class SequenceForAirplanes(Sequence):
    def __init__(self, model, annotation, img_dir, target_size, batch_size, shuffle=True):
        self.model = model
        self.data_sets = []
        if self.model is 'train':
            for e, i in enumerate(os.listdir(annotation)):
                if i.startswith("airplane"):
                    csv_path = os.path.join(annotation, i)
                    img_path = os.path.join(img_dir, i.replace("csv", "jpg"))
                    self.data_sets.append({"csv": csv_path, "jpg": img_path})
        elif self.model is 'val':
            for e, i in enumerate(os.listdir(annotation)):
                if i.startswith("4"):
                    csv_path = os.path.join(annotation, i)
                    img_path = os.path.join(img_dir, i.replace("csv", "jpg"))
                    self.data_sets.append({"csv": csv_path, "jpg": img_path})
        self.image_size = target_size[0:2]
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.data_sets))
        self.shuffle = shuffle

    def __len__(self):
        # 计算每一个epoch的迭代次数
        num_imgs = len(self.data_sets)
        return math.ceil(num_imgs / float(self.batch_size))

    def __getitem__(self, idx):
        # 生成batch_size个索引
        batch_indexs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch = [self.data_sets[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch)
        return X, y

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def read(self, dataset):
        csv_path = dataset['csv']
        jpg_path = dataset['jpg']

        image, image_h, image_w = load_img(path=jpg_path, shape=self.image_size)

        label_matrix = np.zeros([7, 7, 6])
        # 7*7的网格，每格做一次1分类（1种目标）+4偏移量回归+1置信度
        df = pd.read_csv(csv_path)

        for row in df.iterrows():
            xmin = int(row[1][0].split(" ")[0])
            ymin = int(row[1][0].split(" ")[1])
            xmax = int(row[1][0].split(" ")[2])
            ymax = int(row[1][0].split(" ")[3])
            cls = 0
            x = (xmin + xmax) / 2 / image_w
            y = (ymin + ymax) / 2 / image_h
            # 获取归一化中心坐标
            w = (xmax - xmin) / image_w
            h = (ymax - ymin) / image_h
            # 获取归一化高宽
            loc = [7 * x, 7 * y]
            # 找到该中心坐标映射到网格图后相对所在网格的偏移量
            loc_i = int(loc[1])
            loc_j = int(loc[0])
            y = loc[1] - loc_i
            x = loc[0] - loc_j
            # 把中心坐标映射到网格图的所在网格打上对应的分类标签
            if label_matrix[loc_i, loc_j, 5] == 0:
                # 24位为置信度，如果为1表明已有目标由此方格负责预测，则不做进一步处理
                label_matrix[loc_i, loc_j, cls] = 1
                # 对应标签类置1(0到第19共20类)
                label_matrix[loc_i, loc_j, 1:5] = [x, y, w, h]
                # 第20到第23共四位存储目标框信息（偏移量和高宽）
                label_matrix[loc_i, loc_j, 5] = 1
                # 第24位存储置信度，有目标为1，无目标为0

        return image, label_matrix

    def data_generation(self, batch_datasets):
        images = []
        labels = []

        for dataset in batch_datasets:
            image, label = self.read(dataset)
            images.append(image)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)

        # print(
        #     "Person类占该batch总目标个数比例：",
        #     np.count_nonzero(y[:, :, :, 14]) / np.count_nonzero(y[:, :, :, 0:20])
        # )

        return X, y
