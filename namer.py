import os
import sys
import numpy as np


def printnames(d="train"):
    cur_dir = "./dataset/" + d + "/images/"
    cur_imgs = os.listdir(cur_dir)
    with open("train_names.txt", 'a') as fp:
        for elem in cur_imgs:
            fp.write(cur_dir + elem)
            fp.write('\n')


if __name__ == '__main__':
    if(len(sys.argv) == 2):
        printnames(sys.argv[1])
    else:
        print("CANNOT SAVE THE FILE NAMES")
