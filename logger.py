import os
import csv


class Logger:
  def __init__(self, logdir):
    self.logdir = logdir
    self.headers = ["epoch", "train_acc", "val_acc", "train_loss", "val_loss", "avg_grad", "GPU1_tot", "GPU1_occ"]
    self.metrics = [0., 0., 0., 0., 0., 0., 0, 0]
    self.csv_path = '/home/saandeepaath/Desktop/projects/usf/dataviz/D3-pair-project/static/data/metrics.csv'
    with open(self.csv_path, 'w+') as f:
      writer = csv.writer(f)
      writer.writerow(self.headers)
      writer.writerow(self.metrics)
    
  def add_scalar(self, metrics):
    with open(self.csv_path, 'a+') as f:
      writer = csv.writer(f)
      writer.writerow(metrics)


