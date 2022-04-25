import torch 
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os
from sys import exit as e


class CustomDataset(Dataset):
  def __init__(self, args, dtype):
    super().__init__()

    self.args = args
    if dtype == "train":
      self.root_dir = os.path.join(self.args.root, "mnist")
    elif dtype == "test":
      self.root_dir = os.path.join(self.args.root, "mnist_test")

    self.trans = transforms.Compose(
      [
        transforms.ToTensor(),
      ]
      )
    self.cnt_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    self.all_files = []
    self.getAllFiles()
    print(self.cnt_dict)

  def getAllFiles(self):
    for dig in os.listdir(self.root_dir):
      dig_dir = os.path.join(self.root_dir, dig)
      if not os.path.isdir(dig_dir):
        continue
      for fl in os.listdir(dig_dir):
        fpath = os.path.join(dig_dir, fl)
        if os.path.splitext(fpath)[-1] != ".jpg":
          continue
        label = int(fpath.split("/")[-2])
        if self.cnt_dict[label] >= int(self.args.count):
          continue
        self.cnt_dict[label] += 1
        self.all_files.append(fpath)

  def __len__(self):
    return len(self.all_files)

  def __getitem__(self, idx):
    fpath = self.all_files[idx]
    img = self.trans(Image.open(fpath))
    label = int(fpath.split("/")[-2])
    return img, label
  