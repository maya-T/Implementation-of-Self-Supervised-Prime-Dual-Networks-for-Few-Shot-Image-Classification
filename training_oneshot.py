"""#Test"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import os
import numpy as np
from PIL import ImageEnhance
from PIL import Image
import json
from abc import abstractmethod
import torchvision.transforms as transforms
from pathlib import Path
from SSL import *

image_size = 224
base_file = "save_dir/base.json"
eval_file = "save_dir/val.json"
n_shot = 1

n_way = 5
n_query = max(1, int(16* n_way/n_way))
backbone = Backbone()
nb_epoch = 300
lr = 0.01

train_few_shot_params  = dict(n_way = n_way, n_support = n_shot)
base_datamgr  = MyDataManager( image_size, n_query = n_query,  **train_few_shot_params)
val_datamgr  = MyDataManager( image_size, n_query = n_query,  **train_few_shot_params)

base_loader  = base_datamgr.get_data_loader(base_file , aug = True)
val_loader =  val_datamgr.get_data_loader(eval_file , aug = True)


savepath = Path("saved_models/model_prime_1.pch")

backbone = Backbone()
model = Model(n_way, n_shot, n_query, 512, backbone = backbone)

if savepath.is_file():
	with savepath.open("rb") as fp:
		print("Loading saved model")
		model.load_state_dict(torch.load(savepath))

if torch.cuda.is_available():
	print("Using GPU")
	model = model.cuda()

for i in range(1, 10):
	print("Iter {:d}".format(i))
	print("Training prime")
	if i != 1:
		train(model, base_loader, nb_epoch, lr, mode = "Prime")	
		savepath = Path("saved_models/model_prime_"+str(i)+".pch")
		with savepath.open("wb") as fp:
			print("Saving model prime", i)
			torch.save(model.state_dict(),fp)
	print("Training dual")		
	train(model, base_loader, nb_epoch, lr, mode = "Dual")	
	savepath = Path("saved_models/model_dual_"+str(i)+".pch")
	with savepath.open("wb") as fp:
		print("Saving model dual", i)
		torch.save(model.state_dict(),fp)
	print("On base")
	test(model, base_loader)
	print("On val")
	test(model, val_loader)
	
print("training DONE!")




