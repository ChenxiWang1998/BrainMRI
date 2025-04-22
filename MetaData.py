import threading

import numpy as np
import torch.utils.data as tud

MASK_EXIST=False
MASK_NULL=True

class MetaData():
    def __init__(self, name:str, data, mask=None):
        self.name=str(name)
        self.data=list(data)
        if(mask is None):
            mask=np.array([MASK_EXIST] * len(data))
        self.mask=list(np.array(mask))
        assert len(self.data) == len(self.mask)
    
    def __getitem__(self, index):
        return self.data[index], self.mask[index]
    
    def __len__(self):
        return len(self.data)
    
    def splite_by_ratio(self, test_ratio):
        assert test_ratio >= 0 and test_ratio <= 1
        idx=np.random.permutation(len(self))
        cut_idx=int(round(len(self) * test_ratio))
        train_idx, test_idx = idx[cut_idx:], idx[:cut_idx]
        return self.splite_by_idx(train_idx), self.splite_by_idx(test_idx)
    
    def splite_by_idx(self, idx):
        return MetaData(name=self.name, data=[self.data[i] for i in idx], mask=[self.mask[i] for i in idx])
    
    def add(self, src):
        assert type(src) == type(self)
        assert src.name == self.name
        self.data.extend(src.data)
        self.mask.extend(src.mask)
    
    def clone(self):
        return MetaData(self.name, list(self.data), list(self.mask))
    
    def __str__(self):
        return "Meta Data: %s, length:%d" % (self.name, len(self))
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return str(self)

#分类数据
class CategoryMetaData(MetaData):
    def __init__(self, name, idx:list, mask=None):
        super().__init__(name, idx, mask)
        self.idx=idx
    
    #设置数据为one hot向量
    def shift_to_one_hot(self, num=None, alpha=0.0):
        if(num is None):
            num=max(self.idx)+1
        matrix=np.eye(num, dtype=np.float32) * ( 1 - alpha ) + (alpha / num)
        nan_vec=np.zeros(num, dtype=np.float32)
        self.data=[matrix[i] if m == MASK_EXIST else nan_vec for i, m in zip(self.idx, self.mask)]
    
    def splite_by_idx(self, idx):
        data=CategoryMetaData(name=self.name, idx=[self.data[i] for i in idx], mask=[self.mask[i] for i in idx])
        data.idx=[self.idx[i] for i in idx]
        return data
    
    def add(self, src):
        assert isinstance(src, CategoryMetaData)
        super().add(src)
        self.idx.append(src.idx)
    
    def clone(self):
        data = CategoryMetaData(name = self.name, idx = list(self.idx), mask = list(self.mask))
        data.data=list(self.data)
        return data

#核磁数据
class MRIMetaData(MetaData):
    def __init__(self, path, name="mri", mask=None, augment=None):
        super().__init__(name=name, data=path, mask=mask)
        self.augment=augment
        self.path=list(path)
        self.flag=False
    
    def __getitem__(self, index):
        data, mask=super().__getitem__(index)
        if(self.flag ==  False):
            data=np.load(data)
        if(mask == MASK_EXIST and self.augment is not None):
            data=self.augment(data)
        return data, mask
    
    def load(self):
        for i in range(len(self.data)):
            self.data[i]=np.load(self.data[i])
        self.flag=True
    
    def splite_by_idx(self, idx):
        data=MRIMetaData(name=self.name, path=[self.data[i] for i in idx], mask=[self.mask[i] for i in idx], augment=self.augment)
        data.path=[self.path[i] for i in idx]
        data.flag=self.flag
        return data
    
    def add(self, src):
        assert isinstance(src, MRIMetaData)
        assert self.flag == src.flag
        super().add(src)
        self.path.extend(src.path)
    
    def clone(self):
        data = MRIMetaData(path = list(self.path), name = self.name, mask = list(self.mask), augment = self.augment)
        if self.flag:
            data.data=list(self.data)
            data.flag=True
        return data

class MetaDataSet(tud.Dataset):
    def __init__(self, x:list[MetaData], y:list[MetaData]=[], names=None):
        self.x_names=[xx.name for xx in x]
        self.y_names=[yy.name for yy in y]
        self.data=list(x) + list(y)
        self.names=list(names) if names is not None else range(len(self.data[0]))
        self.length = self.__checkLength() #检测全部数据长度是否相等
    
    def __checkLength(self):
        length = len(self.data[0])
        for d in self.data:
            assert length == len(d), "illegal length for dataset:%s(should be %d))" % (d.name, length)
        assert length == len(self.names)
        return length
    
    def __getitem__(self, index):
        r=[]
        for d in self.data:
            r.extend(d[index])
        return self.names[index], r
    
    def __len__(self):
        return self.length
    
    #按照比例划分数据集
    def splite_by_ratio(self, test_ratio):
        from sklearn.model_selection import train_test_split
        train_idx, test_idx=train_test_split([i for i in range(len(self))], test_size=test_ratio)
        return self.splite_by_idx(train_idx), self.splite_by_idx(test_idx)
    
    #按照索引划分数据集
    def splite_by_idx(self, idx):
        return MetaDataSet(
            x = [d.splite_by_idx(idx) for d in self.data[0:len(self.x_names)]], 
            y = [d.splite_by_idx(idx) for d in self.data[len(self.x_names):]],
            names = [self.names[i] for i in idx]
            )
    
    #添加数据集
    def add(self, src):
        assert self.x_names == src.x_names and self.y_names == src.y_names
        for sd, td in zip(self.data, src.data):
            sd.data += td.data
            sd.mask += td.mask
        self.names.append(src.names)
    
    def clone(self):
        data = [d.clone() for d in self.data]
        x = list(data[:len(self.x_names)])
        y = list(data[len(self.y_names):])
        return MetaDataSet(x, y, list(self.names))
    
    def getMetaDataX(self, name = None):
        if isinstance(name, str):
            return self.data[self.x_names.index(name)]
        if name is None:
            name = self.x_names
        data = [ self.getMetaDataX(n) for n in name ]
        return data
    
    def getMetaDataY(self, name = None):
        if isinstance(name, str):
            return self.data[self.y_names.index(name) + len(self.x_names)]
        if name is None:
            name = self.y_names
        data = [ self.getMetaDataY(n) for n in name ]
        return data
    
    @staticmethod
    def concat(*datasets):
        ds=datasets[0]
        data=[dd.clone() for dd in ds.data]
        names=list(ds.names)
        for ds in datasets[1:]:
            for d_des, d_src in zip(data, ds.data):
                d_des.add(d_src)
            names.extend(ds.names)
        return MetaDataSet(x = data[0 : len(ds.x_names)], y = data[len(ds.x_names) : ], names=names)
    
    def __repr__(self):
        s = "dataset x contains:"
        for x in self.data[:len(self.x_names)]:
            s += "\n  " + repr(x)
        s += "\ndataset y contains:"
        for y in self.data[len(self.x_names):]:
            s += "\n  " + repr(y)
        return s

class PreLoader:
    def __init__(self, iter):
        self.iter = iter
        self.result = []
        self.add_task()
    
    def next(self):
        try:
            self.result.append(next(self.iter))
        except StopIteration:
            pass
    
    def add_task(self):
        self.t = threading.Thread(None, self.next)
        self.t.start()
    
    def __iter__(self):
        flag = True
        while flag:
            self.t.join()
            self.t = None
            if(len(self.result) > 0):
                value = self.result.pop()
                self.add_task()
                yield value
            else:
                flag = False

class MetaDataLoader:
    def __init__(self, dataset:MetaDataSet, batch_size=None, shuffle= None, device=None, cache = False, **args):
        self.dataset=dataset
        self.device=device
        self.loader=tud.DataLoader(dataset, batch_size, shuffle, **args)
        
        self.cache = cache
    
    def __iter__(self):
        it = iter(self.loader)
        if self.cache:
            it = PreLoader(it)
            
        for names, data in it:
            data_iter = iter(data)
            x, x_mask=self.parse(self.dataset.x_names, data_iter)
            if(len(self.dataset.y_names) == 0):
                yield names, x, x_mask
            else:
                y, y_mask=self.parse(self.dataset.y_names, data_iter)
                yield names, x, x_mask, y, y_mask
    
    def setDevice(self, device):
        self.device=device
    
    def parse(self, names, d_it):
        input={}
        input_mask={}
        for n in names:
            data = next(d_it)
            mask = next(d_it)
            if self.device is not None:
                data = data.to(self.device)
                mask = mask.to(self.device)
            input.update({n:data.to(self.device)})
            input_mask.update({n:mask.to(self.device)})
        return input, input_mask



