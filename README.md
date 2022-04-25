## **1. SETUP**
```
virtualenv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

## **2. Run code in CLI**
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset mnist --root ~/Projects/fg/fractional_glow/data/ --batch 32 --n_chan 1 --n_class 10 --img_size 28 --iter 100 --lr 1e-4
```
