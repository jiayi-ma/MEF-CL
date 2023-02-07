from __future__ import print_function

import time
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from generate import generate
import scipy.ndimage
import scipy.io as scio

BATCH_SIZE = 18
EPOCHES = 4
LOGGING = 20

MODEL_SAVE_PATH = './models/'

def main():
	print('\nBegin to generate pictures ...\n')
	path = './test_imgs/'
	Format='.JPG'

	T=[]
	for i in range(5):
		index = i + 1
		ue_path = path + 'u' + str(index) + Format
		oe_path = path + 'o' + str(index) + Format

		t=generate(oe_path, ue_path, MODEL_SAVE_PATH, index, output_path = './results/', format=Format)

		T.append(t)
		print("%s time: %s" % (index, t))

if __name__ == '__main__':
	main()
