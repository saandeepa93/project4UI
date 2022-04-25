import argparse

def get_args():
  parser = argparse.ArgumentParser(description="Normalizing Flow")
  parser.add_argument('--dataset', type=str, default='bu', help='Type of dataset')
  parser.add_argument('--root', type=str, default='../../fg/glow_revisited/data/test/', help='Root path of dataset')
  
  parser.add_argument('--test', type=str, default='11', help='Test subject ID')
  parser.add_argument('--count', type=int, default=0, help='Count per class')
  
  parser.add_argument('--img_size', type=int, default=128, help='Image size')
  parser.add_argument('--batch', type=int, default=32, help='Batch size')
  parser.add_argument('--n_chan', type=int, default=3, help='# of channels')
  parser.add_argument('--n_class', type=int, default=6, help='# of class')
  parser.add_argument('--n_block', type=int, default=2, help='# of class')
  
  parser.add_argument('--iter', type=int, default=20, help='# of epochs')
  parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
  
  parser.add_argument('--ckp', type=str, default="", help='Learning rate')

  args = parser.parse_args()
  return args