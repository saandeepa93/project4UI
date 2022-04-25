from operator import delitem
import torch 
from torch import nn 
from torchvision import transforms, utils
import torch.nn.functional as F
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import random 
import numpy as np
import os
import csv

from model import ClassNet
from args import get_args


def seed_everything(seed):
  # random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
  seed_everything(42)
  args = get_args()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())
  csv_path = '/home/saandeepaath/Desktop/projects/usf/dataviz/D3-pair-project/static/data/pred.csv'

  with open(csv_path, 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(["prediction", "label"])
  
  testdataset = torchvision.datasets.MNIST('data', train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                          (0.1307), (0.3081))
                                      ]))
  testloader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch, shuffle=True)

  num = random.randint(0, args.batch)

  images, labels = iter(testloader).next()
  image = images[num]
  label = labels[num]

  print("random", num)

  save_image(image, "/home/saandeepaath/Desktop/projects/usf/dataviz/D3-pair-project/static/data/test.png")
  image = image.unsqueeze(0)
  image = image.to(device)
  print(image.size(), label)

  model = ClassNet(args.n_chan, args.n_block, args.img_size, args.n_class)
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()

  model.load_state_dict(torch.load(args.ckp, map_location=device))

  out = model(image)
  pred = torch.argmax(out, dim=1)

  pred = pred.detach().cpu().item()
  label = label.item()

  with open(csv_path, 'a+') as f:
    writer = csv.writer(f)
    writer.writerow([pred, label])

#   np.savetxt('', pred_numpy.astype(np.int8), delimiter='')
#   torch.save(pred, "/home/saandeepaath/Desktop/projects/usf/dataviz/D3-pair-project/static/data/pred.csv")

#   with open('/home/saandeepaath/Desktop/projects/usf/dataviz/D3-pair-project/static/data/pred.csv', "wb") as file:
#     writer = csv.writer(file, delimiter=',')
#     writer.writerow([pred.item()])


