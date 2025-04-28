
from collections import OrderedDict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchio.transforms import RandomAffine, RandomNoise, Compose

import Data
import Index
import MetaTrainer
import MetaBrain
from Module import *

def load(loader:Data.Loader, folder = "images", file = "T1/T1_brain_linear.npy", baseline = False):
    if baseline:
        loader.index = loader.index.drop_duplicates(subset = "subjectID")
    
    loader = loader.dropna("age") 
    
    mri = loader.loadNpyPath(folder = folder, file = file, filter=True) 
    
    age = loader.loadAge() 
    sex = loader.loadSex(one_hot = True)
    if not DYNAMIC:
        mri.load()
    
    return Data.MetaDataSet(x = [mri, sex], y = [age])

def print_index_dict(index_dict:dict):
    for name, index in index_dict.items():
        print(name, str(index))

class AgeIndex(Index.SingleValueIndex):
    def _calc_index(self, y_pred, y_real):
        from scipy.stats import pearsonr
        
        self.y_pred = y_pred * 10 + 65
        self.y_real = y_real * 10 + 65
        
        RMSE=torch.nn.functional.mse_loss(self.y_pred, self.y_real).sqrt().item()
        MAE = torch.mean(torch.abs(self.y_pred - self.y_real)).item()
        pearson = pearsonr(y_pred.numpy().reshape(-1), y_real.numpy().reshape(-1)).statistic
        
        return [
            Index.IndexValue("RMSE", RMSE, "%.3f"), 
            Index.IndexValue("MAE", MAE, "%.3f"), 
            Index.IndexValue("R", pearson, "%.3f")
        ]

def load_as_CDR(loader : Data.Loader, CDR_stats):
    loader_dict = OrderedDict()
    for cdr in CDR_stats:
        sub_loader = loader.selectCDR(cdr)
        dataset = load(sub_loader)
        dataloader = Data.MetaDataLoader(dataset, batch_size = BATCH_SIZE, device = DEVICE)
        loader_dict.update({cdr:dataloader})
    return loader_dict

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DYNAMIC = True
BATCH_SIZE = 48
DIM_Z = 256
LR = 1e-4
WEIGHT_DECAY = 1e-3

#load train dataset
ADNI_train = load(Data.Loader("ADNI", black = "black.txt", index = "train.csv").selectCDR("CN"))
OASIS = load(Data.Loader("OASIS", black = "black.txt").selectCDR("CN"))
AIBL = load(Data.Loader("AIBL", black = "black.txt").selectCDR("CN"))
HAS = load(Data.Loader("HAS", black = "black.txt").selectCDR("CN"))
IXI = load(Data.Loader("IXI", black = "black.txt").selectCDR("CN"))
train_set = Data.MetaDataSet.concat(ADNI_train, OASIS, AIBL, HAS, IXI)
train_loader = Data.MetaDataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, device = DEVICE, drop_last = True, num_workers = 8)
train_set.getMetaDataX("mri").augment = Compose([RandomAffine(scales = 0, degrees = 30, translation = 12), RandomNoise(std = [0, 0.1])])

dementia_states = ["CN", "MCI", "dementia"]

#ADNI test dataset
test_ADNI = Data.Loader("ADNI", index = "test.csv", black = "black.txt")
test_ADNI.index = test_ADNI.index.drop_duplicates(subset = "subjectID")
test_loader_ADNI_dict = load_as_CDR(test_ADNI, dementia_states)

#CANDI test dataset
test_CANDI = Data.Loader("CANDI", black = "black.txt")
test_CANDI.index = test_CANDI.index.drop_duplicates(subset = "subjectID")
test_loader_CANDI_dict = load_as_CDR(test_CANDI, dementia_states)

model = MetaBrain.MetaBrainViT(dim_z = DIM_Z, layer_num = 6, dropout=[0, 0.1], device = DEVICE) #加载模型
model.addInput(MetaBrain.MetaBrainInput(name = "sex", embedding=generateSexInput(dim_z= DIM_Z), dim_z=DIM_Z)) #性别输出
model.addOutput(MetaBrain.MetaBrainOutput(name = "age", predictor = nn.Linear(DIM_Z, 1))) #年龄预测输出

subtrainers = [
    MetaTrainer.OutputSubTrainer(name = "age", loss_function = F.mse_loss, index_type = AgeIndex)
]

opt = torch.optim.AdamW(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
trainer = MetaTrainer.MetaTrainer(model, subtrainers = subtrainers, opt = opt)

epoch = 0
loss_min = 1e10
n = 0
while n != 10:
    epoch += 1
    print("\nepoch:%s begin(%s)" % (epoch, time.ctime()))
    
    index_train = trainer.calcLoader(train_loader, train = True)
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
    
    if index_train["age"].values_dict["RMSE"] < loss_min:
        loss_min = index_train["age"].values_dict["RMSE"]
        n = 0
    else:
        n += 1
        if n == 5:
            trainer.opt.param_groups[0]["lr"] *= 0.1 #修改学习率为原来的1/10
    
    torch.save(model, "age_%d.pth" % epoch)
    

