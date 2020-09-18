# 1.输入图像预处理，包括尺寸，旋转。
# 2.真实值ground truth变形，shape = (w,h,kp_num) = (224, 224, 24)
# 3.返回一个发生器，用于给模型做输入，以及输出时做损失计算。
import os
import numpy as np
import pandas as pd
import torch
from skimage import io, transform  # 用于图像的IO和变换
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class KeyPointsDataSet(Dataset):
    """服装关键点标记数据集"""

    # 数据集目标化
    def __init__(self, imgF, markF, boxF=None, transform_img=None):
        """
        初始化数据集
        :param imgF: 图像文件清单
        :param markF: 标签文件清单
        :param boxF: 边框文件清单
        """
        self.root = r"E:/Datasets/Landmark_Detect"
        self.exit_box = False
        self.data_imgF = imgF
        self.data_markF = markF
        self.data_boxF = boxF
        self.get_info()

        self.transform_img = transform_img

    def __len__(self):
        return len(self.data_imgF)

    def __getitem__(self, idx):
        H, W = 64.0, 64.0
        try:
            img_id = self.root + "/" + self.data_imgF[idx][0]
            image = io.imread(img_id)
            h, w, c = image.shape
            landmarks = self.string_to_data_list(self.data_markF[idx])
            if self.exit_box:
                bbox = self.string_to_data_list(self.data_boxF[idx])
                # 特别注意，图像尺寸（h,w,c）那么读框的时候应该是先y后x；
                image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                landmarks = self.change_landmarks(bbox, landmarks)
            else:
                landmarks = self.change_landmarks([0, 0, w, h], landmarks)
            h, w, c = image.shape
            image = self.change_img_size(image, H, W)
            landmarks[:, 0] = landmarks[:, 0] * W / w
            landmarks[:, 1] = landmarks[:, 1] * H / h
            # print("new", landmarks)
            if self.transform_img:
                image = self.transform_img(image) / 255.0

            return image, torch.from_numpy(landmarks).reshape(-1, 2)

        except:
            # raise EOFError
            print("error img: ", img_id)
            return self.__getitem__(idx + 1)

    def get_info(self):
        if self.data_boxF:
            self.exit_box = True
            self.data_boxF = self.get_file_info(self.data_boxF)
            self.data_markF = self.get_file_info(self.data_markF)
            self.data_imgF = self.get_file_info(self.data_imgF)
            return
        else:
            self.data_markF = self.get_file_info(self.data_markF)
            self.data_imgF = self.get_file_info(self.data_imgF)
            return

    @staticmethod
    def get_file_info(file_path):
        file_info = pd.read_csv(file_path)
        return file_info.values

    @staticmethod
    def string_to_data_list(line):
        index = 0
        value = line[0].split()
        for var in value:
            value[index] = int(var)
            index += 1
        return value

    @staticmethod
    def change_landmarks(bbox, landmarks):
        x_1, y_1, x_2, y_2 = bbox
        keyValue_x = []
        keyValue_y = []
        for i in range(len(landmarks)):
            if i % 3 == 0:
                continue
            elif i % 3 == 1:
                temp = landmarks[i] - x_1
                if temp < 0:
                    temp = 0
                keyValue_x.append(temp)
            else:
                temp_y = landmarks[i] - y_1
                if temp_y < 0:
                    temp_y = 0
                keyValue_y.append(temp_y)
        return np.array(list(zip(keyValue_x, keyValue_y)))

    @staticmethod
    def change_img_size(image, h, w):
        return transform.resize(image, (h, w))


class ToTensor(object):
    """将样本中的ndarrays转换为Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)


transform_img = transforms.Compose([
    transforms.ToTensor(),  # 将图像(Image)转成Tensor,归一化[0,1]
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 将tensor标准化[-1,1]
])
transform_heat = transforms.Compose([
    ToTensor(),  # 将图像(Image)转成Tensor,归一化[0,1]
])

data_file = r"E:\Datasets\Landmark_Detect\Anno"
img_file = r"\train.txt"  # val.txt or test.txt
mark_file = r"\train_landmarks.txt"  # val_landmarks.txt or test_landmarks.txt
box_file = r"\train_bbox.txt"  # val_bbox.txt or test_bbox.txt

fashionDataset = KeyPointsDataSet(imgF=data_file + img_file,
                                  markF=data_file + mark_file,
                                  boxF=data_file + box_file,
                                  transform_img=transform_img,
                                  )
dataloader = DataLoader(dataset=fashionDataset, batch_size=2, shuffle=True)

if __name__ == "__main__":
    # 如果图片的大小不同，则无法构成batch
    for i_batch, data in enumerate(dataloader):
        img, landmarks = data
        # img = img.transpose((0, 2, 3, 1))
        print(img.shape, landmarks.shape)
