import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchio.transforms import RandomAffine, RandomNoise, Compose

import Data
import Index
import MetaTrainer
import MetaBrain
from Module import *

def load(loader:Data.Loader, folder="images", file="T1/T1_brain_linear.npy"):
    
    loader = loader.dropna("SUVR")
    
    age=loader.loadAge() #加载年龄数据
    sex=loader.loadSex(one_hot=True) #加载性别数据
    apoe=loader.loadAPOE(one_hot=True) #加载APOE数据
    
    suvr=loader.loadMetaData("SUVR")
    suvr.data = [ np.log(d / 1.11) for d in suvr.data]
    
    suvrc = Data.CategoryMetaData("SUVRc", idx = [ 0 if v < 0 else 1 for v in suvr.data ])
    suvrc.shift_to_one_hot(2, ALPHA)
    
    # 加载核磁数据
    mri=loader.loadNpyPath(folder=folder, file=file) #核磁路径数组
    if not DYNAMIC:
        mri.load()
    
    dataset=Data.MetaDataSet(x=[mri, sex, age, apoe], y=[suvr, suvrc], names = loader.index.index.tolist())
    
    return dataset

def print_index_dict(index_dict:dict):
    for name, index in index_dict.items():
        print(name, str(index))

DYNAMIC = True
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ALPHA = 0
BATCH_SIZE = 48
LR = 1e-4
WEIGHT_DECAY = 1e-3
DIM_Z = 256

#load ADNI train dataset
ADNI_train = Data.Loader("ADNI", index="train.csv", black="black.txt")
ADNI_train = ADNI_train.dropna(["SUVR"])
train_set=load(ADNI_train)
train_loader=Data.MetaDataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, drop_last = True, num_workers = 8)
train_set.getMetaDataX("mri").augment=Compose([RandomAffine(scales=0, degrees=30, translation = 12), RandomNoise(std=[0, 0.1])])

#load ADNI test dataset
ADNI=Data.Loader("ADNI", index="test.csv", black="black.txt")
ADNI = ADNI.dropna(["APOE", "SUVR"])
ADNI.index=ADNI.index.iloc[np.random.permutation(len(ADNI.index))].drop_duplicates("subjectID") #随机选取随访中的一次数据
ADNI.index.to_csv("SUVR/test(ADNI).csv")
test_loader_ADNI_CN=Data.MetaDataLoader(load(ADNI.selectCDR("CN")), batch_size = BATCH_SIZE)
test_loader_ADNI_MCI=Data.MetaDataLoader(load(ADNI.selectCDR("MCI")), batch_size = BATCH_SIZE)
test_loader_ADNI_dementia=Data.MetaDataLoader(load(ADNI.selectCDR("dementia")), batch_size = BATCH_SIZE)
test_dataset_ADNI = Data.MetaDataSet.concat(test_loader_ADNI_CN.dataset, test_loader_ADNI_MCI.dataset, test_loader_ADNI_dementia.dataset)
test_loader_ADNI = Data.MetaDataLoader(test_dataset_ADNI, batch_size = BATCH_SIZE, num_workers = 8)
test_loader_ADNI_dict = {
    "total":test_loader_ADNI, 
    "CN" : test_loader_ADNI_CN, 
    "MCI" : test_loader_ADNI_MCI, 
    "dementia" : test_loader_ADNI_dementia, 
}

#load CANDI test dataset
CANDI=Data.Loader("CANDI", index="index2.csv", black="black.txt")
CANDI.index.to_csv("SUVR/test(CANDI).csv")
CANDI = CANDI.dropna(["APOE", "SUVR"])
test_loader_CANDI_CN=Data.MetaDataLoader(load(CANDI.selectCDR("CN")), batch_size = BATCH_SIZE)
test_loader_CANDI_MCI=Data.MetaDataLoader(load(CANDI.selectCDR("MCI")), batch_size = BATCH_SIZE)
test_loader_CANDI_dementia=Data.MetaDataLoader(load(CANDI.selectCDR("dementia")), batch_size = BATCH_SIZE)
test_dataset_CANDI = Data.MetaDataSet.concat(test_loader_CANDI_MCI.dataset, test_loader_CANDI_dementia.dataset)
test_loader_CANDI = Data.MetaDataLoader(test_dataset_CANDI, batch_size = BATCH_SIZE, num_workers = 8)
test_loader_CANDI_dict = {
    "total":test_loader_CANDI, 
    "CN" : test_loader_CANDI_CN, 
    "MCI" : test_loader_CANDI_MCI, 
    "dementia" : test_loader_CANDI_dementia, 
}

#generate model
model:MetaBrain.MetaBrainViT = torch.load("CDR.pth")
model.clearOutput()
model.addInput(MetaBrain.MetaBrainInput(name="APOE", embedding=generateAPOEInput(), dim_z=DIM_Z))
model.addOutput(MetaBrain.MetaBrainOutput(name="SUVR", predictor=nn.Sequential(
    LinearLayer(DIM_Z, DIM_Z), 
    nn.Linear(DIM_Z, 1)
)))

subtrainers=[
    MetaTrainer.OutputSubTrainer("SUVR", loss_function=F.mse_loss, index_type=Index.SingleValueCategoryIndex), 
]

trainer=MetaTrainer.MetaTrainer(model, subtrainers, opt=torch.optim.AdamW(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY))

epoch = 0
loss_min = 1e10
n = 0
while n != 10:
    epoch += 1
    print("\nepoch:%s begin(%s)" % (epoch, time.ctime()))

    index_train = trainer.calcLoader(train_loader, train = True)
    index_train=trainer.calcLoader(train_loader, train=True)
    print("train set")
    print_index_dict(index_train)
    
    for cog, loader in test_loader_ADNI_dict.items():
        index_dict=trainer.calcLoader(loader, train = False)
        print("ADNI test set(%s):" % cog)
        print_index_dict(index_dict)
    
    for cog, loader in test_loader_CANDI_dict.items():
        index_dict=trainer.calcLoader(loader, train = False)
        print("CANDI test set(%s):" % cog)
        print_index_dict(index_dict)
    
    if index_train["SUVR"].values_dict["RMSE"] < loss_min:
        loss_min = index_train["SUVR"].values_dict["RMSE"]
        min_epoch = 0
    else:
        min_epoch += 1
        if min_epoch == 5:
            trainer.opt.param_groups[0]["lr"] *= 0.1
    torch.save(model, "CDR_%d.pth" % epoch)
    