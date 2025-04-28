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

dynamic=True

def load(loader:Data.Loader, folder="images", file="T1/T1_brain_linear.npy"):
    loader = loader.dropna("CDR")
    
    age=loader.loadAge()
    sex=loader.loadSex(one_hot=True)
    
    #加载认知数据
    cdr=loader.loadMetaData("CDR")
    cdr_groups=["CN" if v==0 else ( "MCI" if v == 0.5 else "dementia" ) for v in np.array(cdr.data).reshape(-1)]
    cdr_CN_vs_MCI=Data.CategoryMetaData("CDR_CN_vs_MCI", *Data.searchIndex(cat=["CN", "MCI"], data=cdr_groups, mask=cdr.mask))
    cdr_CN_vs_MCI.shift_to_one_hot(2, alpha = ALPHA)
    cdr_CN_vs_dementia=Data.CategoryMetaData("CDR_CN_vs_dementia", *Data.searchIndex(cat=["CN", "dementia"], data=cdr_groups, mask=cdr.mask))
    cdr_CN_vs_dementia.shift_to_one_hot(2, alpha = ALPHA)
    cdr_MCI_vs_dementia=Data.CategoryMetaData("CDR_MCI_vs_dementia", *Data.searchIndex(cat=["MCI", "dementia"], data=cdr_groups, mask=cdr.mask))
    cdr_MCI_vs_dementia.shift_to_one_hot(2, alpha = ALPHA)
    cdr_CN_vs_MCI_vs_dementia=Data.CategoryMetaData("CDR_CN_vs_MCI_vs_dementia", *Data.searchIndex(cat=["CN", "MCI", "dementia"], data=cdr_groups, mask=cdr.mask))
    cdr_CN_vs_MCI_vs_dementia.shift_to_one_hot(3, alpha = ALPHA)
    
    mri=loader.loadNpyPath(folder=folder, file=file)
    if not dynamic:
        mri.load()
    
    dataset=Data.MetaDataSet(x=[mri, sex, age], y=[cdr, cdr_CN_vs_MCI, cdr_CN_vs_dementia, cdr_MCI_vs_dementia, cdr_CN_vs_MCI_vs_dementia], names = loader.index.index)
    
    return dataset

def print_index_dict(index_dict:dict):
    for name, index in index_dict.items():
        print(name, str(index))

def generate_model():
    model:MetaBrain.MetaBrainViT=torch.load("age.pth")
    model.clearOutput()
    model.addInput(MetaBrain.MetaBrainInput(name="age", embedding=generateAgeInput(dim_z = DIM_Z), dim_z=DIM_Z))
    classifier=nn.Sequential(
        LinearLayer(DIM_Z, DIM_Z), 
        nn.Linear(DIM_Z, 3)
    )
    model.addOutput(MetaBrain.MetaBrainOutput(name="CDR_CN_vs_MCI_vs_dementia", predictor=classifier))
    model.addOutput(MetaBrain.MetaBrainOutput(name="CDR_CN_vs_MCI", predictor=nn.Sequential(
        classifier, 
        IndexSelector(0, 1)
    )))
    model.addOutput(MetaBrain.MetaBrainOutput(name="CDR_CN_vs_dementia", predictor=nn.Sequential(
        classifier, 
        IndexSelector(0, 2)
    )))
    model.addOutput(MetaBrain.MetaBrainOutput(name="CDR_MCI_vs_dementia", predictor=nn.Sequential(
        classifier, 
        IndexSelector(1, 2)
    )))
    model.addOutput(MetaBrain.MetaBrainOutput(name="CDR", predictor=nn.Sequential(
        LinearLayer(DIM_Z, DIM_Z), 
        nn.Linear(DIM_Z, 1)
    )))
    return model

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DYNAMIC = True
ALPHA = 0
BATCH_SIZE = 48
DIM_Z = 256
LR = 1e-4
WEIGHT_DECAY = 1e-3


#加载训练数据
ADNI_train = load(Data.Loader("ADNI", black="black.txt", index="train.csv"))
OASIS = load(Data.Loader("OASIS", black="black.txt"))
AIBL = load(Data.Loader("AIBL", black="black.txt"))
train_set = Data.MetaDataSet.concat(ADNI_train, OASIS, AIBL)
train_loader=Data.MetaDataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True, drop_last=True, device=DEVICE, num_workers = 12)

train_set.getMetaDataX("mri").augment=Compose([RandomAffine(scales=0, degrees=30, translation=12), RandomNoise(std=[0, 0.1])])

#加载CANDI测试集
test_CANDI = Data.Loader("CANDI", black="black.txt")
test_CANDI.index = test_CANDI.index.drop_duplicates(subset="subjectID")
testset_CANDI=load(test_CANDI)
test_loader_CANDI=Data.MetaDataLoader(testset_CANDI, batch_size = BATCH_SIZE, device=DEVICE, num_workers = 8)

#加载ADNI测试集
test_ADNI = Data.Loader("ADNI", index="test.csv", black="black.txt")
test_ADNI.index = test_ADNI.index.iloc[np.random.permutation(len(test_ADNI.index))]
test_ADNI.index = test_ADNI.index.drop_duplicates(subset="subjectID")
testset_ADNI=load(test_ADNI)
test_loader_ADNI=Data.MetaDataLoader(testset_ADNI, batch_size = BATCH_SIZE, device=DEVICE, num_workers = 8)

model = generate_model()

subtrainers=[
    MetaTrainer.OutputSubTrainer("CDR_CN_vs_MCI", loss_function=F.cross_entropy, index_type=Index.BinaryCatogoryIndex), 
    MetaTrainer.OutputSubTrainer("CDR_CN_vs_dementia", loss_function=F.cross_entropy, index_type=Index.BinaryCatogoryIndex), 
    MetaTrainer.OutputSubTrainer("CDR_MCI_vs_dementia", loss_function=F.cross_entropy, index_type=Index.BinaryCatogoryIndex),
    MetaTrainer.OutputSubTrainer("CDR_CN_vs_MCI_vs_dementia", loss_function=F.cross_entropy, index_type=Index.CatogoryIndex),
    MetaTrainer.OutputSubTrainer("CDR", loss_function=F.mse_loss, index_type=Index.SingleValueIndex),
]

trainer=MetaTrainer.MetaTrainer(model, subtrainers, opt = torch.optim.AdamW(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY))

painters_dict={
    "CDR_CN_vs_MCI" : Index.IndexPainter(["ACC", "AUC"]), 
    "CDR_CN_vs_dementia" : Index.IndexPainter(["ACC", "AUC"]), 
    "CDR_MCI_vs_dementia" : Index.IndexPainter(["ACC", "AUC"]), 
    "CDR_CN_vs_MCI_vs_dementia" : Index.IndexPainter(["ACC"]), 
    "CDR" : Index.IndexPainter(["RMSE"])
}

epoch = 0
loss_min = 1e10
n = 0
while n != 10:
    epoch += 1
    print("\nepoch:%s begin(%s)" % (epoch, time.ctime()))

    index_train=trainer.calcLoader(train_loader, train=True)
    index_train=trainer.calcLoader(train_loader, train=True)
    print("train set")
    print_index_dict(index_train)
    
    index_dict=trainer.calcLoader(test_loader_ADNI, train = False)
    print("ADNI test set:")
    print_index_dict(index_dict)
    
    index_dict=trainer.calcLoader(test_loader_CANDI, train = False)
    print("CANDI test set:")
    print_index_dict(index_dict)
    
    if index_train["CDR"].values_dict["RMSE"] < loss_min:
        loss_min = index_train["CDR"].values_dict["RMSE"]
        min_epoch = 0
    else:
        min_epoch += 1
        if min_epoch == 5:
            trainer.opt.param_groups[0]["lr"] *= 0.1 
            print("reset learn rate as:", trainer.opt.param_groups[0]["lr"])
    torch.save(model, "CDR_%d.pth" % epoch) 
    
