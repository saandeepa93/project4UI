import torch 
import numpy as np
from torch import nn, optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pynvml import *
from tqdm import tqdm
import random
import os
from sklearn.metrics import accuracy_score
import subprocess
from sys import exit as e

from args import get_args
from dataset import CustomDataset
from model import ClassNet
from logger import Logger

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def compute_avg_grad(model):
  avg_grad = 0
  cnt = 0
  for name, param in model.named_parameters():
    avg_grad += param.grad.abs().mean().item()
    cnt += 1
  avg_grad /= cnt
  return avg_grad

def seed_everything(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True

def get_loader(args, dtype):
  if args.dataset == "mnist":
    train_dataset = datasets.MNIST(root = 'data', train = True, transform = transforms.ToTensor(), download = True)
    test_dataset = datasets.MNIST(root = 'data', train = False, transform = transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=1),
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=True, num_workers=1),

    # dataset = CustomDataset(args, dtype)
  # loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    return trainloader, testloader

def get_valScores(testloader, model):
  model.eval()
  y_pred = []
  y_true= []
  val_loss_lst = 0
  for b, (x, target) in enumerate(testloader, 0):
    x = x.to(device)
    target = target.to(device)
    with torch.no_grad():
      out = model(x)
      val_loss = criterion(out, target)
      val_loss_lst += val_loss.item()

      y_true += target.tolist()
      y_pred += torch.argmax(out, dim=1).tolist()

    # y_true.append(target[0].item())
    # y_pred.append(torch.argmax(out, dim=1)[0].item())

  return accuracy_score(y_pred, y_true), val_loss_lst/len(testloader)


if __name__ == "__main__":
  seed_everything(42)
  args = get_args()
  writer = Logger("./logs")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  traindataset = torchvision.datasets.MNIST('data', train=True, download=True,
                                      transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                          (0.1307), (0.3081))
                                      ]))
  index = [k for k in range(20000)]
  traindataset = torch.utils.data.Subset(traindataset, index)

  trainloader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch, shuffle=True)

  testdataset = torchvision.datasets.MNIST('data', train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                          (0.1307), (0.3081))
                                      ]))
  testloader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch, shuffle=True)
  

  model = ClassNet(args.n_chan, args.n_block, args.img_size, args.n_class)
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=args.lr)

  
  nvmlInit()
  h = nvmlDeviceGetHandleByIndex(1)
  info = nvmlDeviceGetMemoryInfo(h)
  gpu_tot = info.total//1e+6

  pbar = tqdm(range(args.iter))
  for epoch in pbar:
    y_true_train, y_pred_train = [], []
    for b, (x, label) in enumerate(trainloader, 0):
      x = x.to(device)
      label = label.to(device)
      out = model(x)

      loss = criterion(out, label)
      model.zero_grad()
      loss.backward()
      optimizer.step()

      with torch.no_grad():
        y_true_train += label.tolist()
        y_pred_train += torch.argmax(out, dim=1).tolist()
    
    pbar.set_description("Getting validation scores")
    with torch.no_grad():
      train_acc = accuracy_score(y_pred_train, y_true_train)
      val_acc, val_loss = get_valScores(testloader, model)
    
    avg_grad = compute_avg_grad(model)
    gpu_occ = get_gpu_memory_map()[1]
    
    metrics = [epoch, round(train_acc, 3), round(val_acc, 3), round(loss.item(), 3), round(val_loss, 3), \
      round(avg_grad, 4), gpu_tot, gpu_occ]
    writer.add_scalar(metrics)

    if epoch % 10 == 0:
      pbar.set_description("saving model..")
      torch.save(model.state_dict(), f"./checkpoint/model_{str(epoch+1).zfill(3)}.pt")

    pbar.set_description(f"epoch: {epoch}; {get_gpu_memory_map()[1]}; Train loss: {round(loss.item(), 3)}; Val Acc: {round(val_acc, 3)}; Train_acc: {round(train_acc, 3)}")
  
