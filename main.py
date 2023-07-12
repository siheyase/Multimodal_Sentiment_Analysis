import torch
import numpy as np
import pandas as pd
from model import train, test
import argparse
import warnings

warnings.filterwarnings("ignore")


# 解析命令行
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_only', action='store_true', help='use text only')
    parser.add_argument('--image_only', action='store_true', help='use image only')
    parser.add_argument('--train', action='store_true', help='train model')
    parser.add_argument('--test', action='store_true', help='predict test result')
    parser.add_argument('--model', default=3, help='model1:gru+resnet18 '
                                                   'model2:gru+VGG16 '
                                                   'model3:gru(word2Vec)+ResNet16', type=int)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--epochs', default=10, type=int)

    return parser.parse_args()


args = get_args()
print('args:', args)

assert ((args.text_only and args.image_only) == False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 27
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.deterministic = True

if __name__ == "__main__":
    if args.train:
        train(args)
    if args.test:
        test(args)

