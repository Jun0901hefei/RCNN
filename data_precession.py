import os
import pandas as pd
import torchvision
import torch
from torch.utils.data import Dataset,DataLoader
def read_data_bananas(is_train=True):
    #读取label的csv文件的地址
    label_csv= os.path.join("./data/banana-detection", 'bananas_train' if is_train
    else 'bananas_val', 'label.csv')
    #读取csv文件并将'img_name'设置为index
    csv_data = pd.read_csv(label_csv).set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join("./data/banana-detection", 'bananas_train' if is_train else
            'bananas_val', 'images', f'{img_name}')))
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的香蕉类（索引为0）
        targets.append(list(target))
    return images, torch.tensor(targets)/ 256
class BananasDataset(Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))
    def __getitem__(self, idx):
        return self.features[idx].float(), self.labels[idx]
    def __len__(self):
        return len(self.features)
def load_data_bananas(batch_size):
    """dataloader"""
    train_iter = DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    test_iter = DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, test_iter
if __name__ == '__main__':
    import show_boundingbox
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_bananas(batch_size)
    batch = next(iter(train_iter))
    imgs = (batch[0][0:10].permute(0, 2, 3, 1)) / 255
    show_boundingbox.draw_bbox_on_image(imgs,batch[1][0:10,1:5]*edge_size,nrows=2,ncols=5)