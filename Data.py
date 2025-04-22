import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
import torchio
from torchio.transforms import Compose, RescaleIntensity, CropOrPad

from MetaData import *

DEVICE=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
LABELS=["AD", "SMC", "MCI", "LMCI", "EMCI", "CN", "nonAD"]

def searchIndex(cat, data, mask):
    r_idx=[]
    r_mask=[]
    for g, m in zip(data, mask):
        if m == MASK_EXIST and g in cat:
            r_idx.append(cat.index(g))
            r_mask.append(MASK_EXIST)
        else:
            r_idx.append(0)
            r_mask.append(MASK_NULL)
    return r_idx, r_mask

class Dataset(tud.Dataset):
    def __init__(self, *data):
        self.data=data
        self.length=min([len(d) for d in data])
        for d in data:
            assert self.length == len(d)
    
    def __getitem__(self, index):
        return [d[index] for d in self.data]
    
    def __len__(self):
        return self.length

class Loader():
    def __init__(self, path, index="index.csv", black=None):
        self.path=path
        if isinstance(index, str):
            if not index.startswith("/"):
                index=self.path+"/"+index
            index=pd.read_csv(index, index_col=0)
        assert isinstance(index, pd.DataFrame)
        self.index=index
        if(black is not None):
            with open(self.path+"/black.txt") as f:
                black_names=[n[:-1] for n in f.readlines()]
                black_names=[b for b in black_names if b in self.index.index]
            self.drop(black_names)
    
    def dropna(self, subset):
        return Loader(self.path, self.index.dropna(subset=subset))
    
    def retain(self, names):
        return Loader(self.path, self.index.dropna(subset=self.index.loc[[n in names for n in self.index]]))
    
    def drop(self, names):
        names=[n for n in names if n in self.index.index]
        self.index=self.index.drop(names, axis=0)
    
    def selectIndex(self, name, values:list):
        assert name in self.index.columns
        if not isinstance(values, list) and not isinstance(values, tuple):
            values=[values]
        return Loader(self.path, self.index.loc[[v in values for v in self.index[name]]])
    
    def selectGroup(self, values):
        if isinstance(values, str):
            values = [values]
        G=set(self.index["group"])
        for v in values:
            assert v in G, "no type:%s in LABELS(%s)" % (v, G)
        return self.selectIndex("group", values)
    
    def selectCDR(self, CDRtype=None):
        values=set()
        assert CDRtype is not None
        if not isinstance(CDRtype, list):
            CDRtype=[CDRtype]
        for t in CDRtype:
            if isinstance(t, str):
                assert t in ["CN", "MCI", "dementia"]
                if(t == "CN"):
                    values.add(0.0)
                elif(t == "MCI"):
                    values.add(0.5)
                elif(t == "dementia"):
                    values.update([1.0, 2.0, 3.0])
            else:
                values.add(float(t))
        return Loader(self.path, self.index.loc[[cdr in values for cdr in self.index["CDR"]]])
    
    def selectBaseline(self):
        return Loader(self.path, self.index.drop_duplicates(subset="subjectID"))
    
    def _load(self, reader, folder, file, filter=False, resize=None, rescale=None):
        niix_path=self.path+"/"+folder
        files_names=os.listdir(niix_path)
        index_names=self.index.index.tolist()
        names=[n for n in index_names if n in files_names]
        if(filter):
            self.index=self.index.loc[names]
        img=[reader("%s/%s/%s" % (niix_path, n, file)) for n in names]
        if(resize is not None or rescale is not None):
            for i in range(len(img)):
                transform_list=[]
                if(rescale is not None):
                    transform_list.append(RescaleIntensity((0, 1), (0, 99)))
                if(resize is not None):
                    transform_list.append(CropOrPad(resize))
                compose=Compose(transform_list)
                img[i]=compose(img[i])
        return img
    
    def loadNiix(self, folder, file, filter=False, resize=None, rescale=None):
        def reader(path):
            return torchio.ScalarImage(path).data
        return self._load(reader, folder, file, filter, resize, rescale)
    
    def loadNpy(self, folder, file, filter=False, resize=None, rescale=None):
        def reader(path):
            return torch.Tensor(np.load(path))
        return self._load(reader, folder, file, filter, resize, rescale)
    
    def loadNpyPath(self, folder, file, filter=False):
        niix_path=self.path+"/"+folder
        names=[n for n in self.index.index if n in os.listdir(niix_path)]
        if(filter):
            self.index=self.index.loc[names]
        path = ["%s/%s/%s" % (niix_path, n, file) for n in names]
        path = [ p for p in path if os.path.exists(p)]
        return MRIMetaData(path)
    
    def loadIndex(self, name, z_score=False, log=False):
        data=self.index[name]
        mask = data.isna()
        data=np.array(data)
        if(log):
            data=np.log(data)
        if(z_score):
            data_masked=data[np.logical_not(mask)]
            data=(data - np.mean(data_masked)) / np.std(data_masked)
        return data, mask
    
    def loadMetaData(self, name, z_score=False, log=False, fillna = 0):
        data, mask = self.loadIndex(name=name, z_score=z_score, log=log)
        data[mask] = fillna
        data = list(data.astype("f4").reshape((-1, 1)))
        return MetaData(name, data = data, mask = mask)
    
    def loadAge(self, norm=[65, 10]):
        age=self.loadMetaData("age")
        if (norm):
            age.data=[ (d - norm[0]) / norm[1] for d in age.data]
        return age
    
    def loadGroup(self, one_hot=False, alpha=0.0, name="group", cat=None):
        group, group_mask=self.loadIndex("group")
        if(cat is None):
            cat=list(set(group))
        group_idx, group_mask=searchIndex(cat=cat, data=group, mask=group_mask)
        data=CategoryMetaData(name, idx=group_idx, mask=group_mask)
        if(one_hot):
            data.shift_to_one_hot(num = len(cat), alpha = alpha)
        return data
    
    def loadSex(self, one_hot=False, alpha=0.0, name="sex", cat=["M", "F"]):
        sex, sex_mask=self.loadIndex("sex")
        sex_idx, sex_mask=searchIndex(cat=cat, data=sex, mask=sex_mask)
        data=CategoryMetaData(name, idx=sex_idx, mask=sex_mask)
        if(one_hot):
            data.shift_to_one_hot(num = len(cat), alpha = alpha)
        return data
    
    def loadAPOE(self, one_hot=False, alpha=0.0, name="APOE", cat=[22, 23, 24, 33, 34, 44]):
        apoe, apoe_mask=self.loadIndex("APOE")
        apoe_idx, apoe_mask=searchIndex(cat=[22, 23, 24, 33, 34, 44], data=apoe, mask=apoe_mask)
        data=CategoryMetaData(name, idx=apoe_idx, mask=apoe_mask)
        if(one_hot):
            data.shift_to_one_hot(num = len(cat), alpha = alpha)
        return data
    
    def loadCDR(self):
        return self.loadMetaData("CDR")
    
    def loadStats(self, aparc, meas, hemi=None, folder="stats", log=False, norm=False, name=None, filter=False):
        path=self.path + "/" + folder + "/"
        def load_partial(hemi):
            data=pd.read_csv(path+"%s.%s.%s.csv" % (hemi, aparc, meas), index_col=0)
            col_global=[]
            col_partial=[]
            for c in data.columns:
                if c.startswith(hemi):
                    col_partial.append(c)
                else:
                    col_global.append(c)
            return data[col_partial], data[col_global]
        if hemi is None:
            left, left_global=load_partial("lh")
            right, _=load_partial("rh")
            data=[left, right, left_global]
        else:
            assert hemi in ["lh", "rh"]
            data_partial, data_global=load_partial(hemi)
            data=[data_partial, data_global]
        data=pd.concat(data, axis=1).astype("f4")
        names=[ n  for n in data.index if n in self.index.index]
        data=data.loc[names]
        if(filter):
            self.index=self.index.loc[names]
        if log:
            data=np.log(data)
        if norm:
            data=(data - data.mean()) / data.std()
        if name is None:
            name=""
            if(hemi is not None):
                name += hemi+"_"
            name += aparc + "_" + meas
            name:str=name
            name = name.replace(".", "_")
        return MetaData(name=name, data=list(data.values))
    
    def __len__(self):
        return len(self.index)
    
    @staticmethod
    def _searchIndex(cat, data, mask):
        r_idx=[]
        r_mask=[]
        for g, m in zip(data, mask):
            if(m == MASK_EXIST):
                if(g in cat):
                    r_idx.append(cat.index(g))
                else:
                    r_idx.append(0)
        return r_idx, r_mask
    
    def clone(self):
        return Loader(self.path, self.index)
    
    def __repr__(self):
        s="path:"+self.path+"\n"
        s+=repr(self.index)
        return s

def parse_date(date):
    d=date.split("-")
    return datetime.date(*[int(v) for v in d])

def splite_loader(loader, mean = 4, interval = 0.5):
    index = loader.index
    current_index_list = []
    future_index_list = []
    for s in set(index["subjectID"]):
        sindex = index[index["subjectID"] == s]
        date = [parse_date(d) for d in sindex["date"]]
        intervals = np.array([ (d - date[0]).days / 365.25 for d in date])
        intervals = np.abs(intervals - mean)
        min_idx = np.argmin(intervals)
        if intervals[min_idx] < interval:
            current_index_list.append(sindex.iloc[0])
            future_index_list.append(sindex.iloc[min_idx])
    current_loader = Loader(loader.path, index = pd.DataFrame(current_index_list))
    future_loader = Loader(loader.path, index = pd.DataFrame(future_index_list))
    return current_loader, future_loader

COLOR={
    "CN":"#00c000",
    "MCI":"#ffc000",
    "dementia":"#ff0000",
    "train":"#ff0000",
    "test(ADNI)":"#00ff00",
    "test(CANDI)":"#0000ff",
}

def parameters_stats(model : torch.nn.Module):
    num = 0
    for p in model.parameters():
        num += p.numel()
    return num


