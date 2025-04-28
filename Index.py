from abc import ABC, abstractmethod
from math import sqrt
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score
import torch

class Draw:
    def __init__(self, path, title="", legend=False):
        self.path=path
        self.title=title
        self.legend=legend
    
    def __enter__(self):
        plt.figure(figsize=(6, 6)) 
        plt.title(self.title)
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if(self.legend):
            plt.legend()
        plt.savefig(self.path)
        plt.close()

class DrawROC:
    def __init__(self, path, title="", legend=False):
        self.path=path
        self.title=title
        self.legend=legend
    
    def __enter__(self):
        plt.figure(figsize=(6, 6)) 
        plt.title(self.title)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
    
    def __exit__(self, exc_type, exc_value, exc_tb):
        if(self.legend):
            plt.legend()
        plt.savefig(self.path)
        plt.close()

class IndexPainter():
    def __init__(self, targets) -> None:
        self.index_list_dict=dict()
        self.setTargets(targets)
    
    def setTargets(self, targets):
        self.targets=targets
    
    def add(self, name, index):
        if name not in self.index_list_dict.keys():
            self.index_list_dict.update({name:list()})
        self.index_list_dict[name].append(index)
    
    def savefig(self, folder, colors=None):
        if not os.path.exists(folder):
            os.makedirs(folder)
        for t in self.targets:
            values_list = []
            with Painter.Draw(path = "%s/%s.pdf" % (folder, t), title = t, legend=True):
                for dataset_name, index_list in self.index_list_dict.items():
                    values=pd.Series([i.values_dict[t] for i in index_list], index = range(1, 1+len(index_list)), name = dataset_name)
                    c=None if colors is None else colors[dataset_name]
                    plt.plot(values.index, values, label=values.name, c = c)
                    values_list.append(values)
            data = pd.DataFrame(values_list).transpose()
            data.to_csv("%s/%s.csv" % (folder, t))

class IndexValue():
    def __init__(self, name, value, format = None):
        self.name = name
        self.value = value
        self.format = format
    
    def __str__(self):
        return self.name + ":" + (str(self.value) if self.format is None else self.format % self.value)

class IndexBase(ABC):
    def __init__(self):
        self.calculated = False

        self.y_pred_list = [] #predicted value
        self.y_real_list = [] #real value
        self.names_list = [] #sample id
    
    def record(self, y_pred, y_real, names=None):
        self.y_pred_list.append(y_pred.detach().cpu())
        self.y_real_list.append(y_real.detach().cpu())
        if names is not None:
            self.names_list.append(names)
        self.calculated=False
    
    def calc(self):
        if not self.calculated:
            if len(self.y_pred_list) == 0:
                return False
            self.y_pred=torch.concat(self.y_pred_list, dim=0)
            self.y_real=torch.concat(self.y_real_list, dim=0)
            if len(self.names_list) > 0:
                self.names = []
                for n in self.names_list:
                    self.names.extend(n)
            try:
                self.values = self._calc_index(self.y_pred, self.y_real)
                self.values_dict = dict([v.name, v.value] for v in self.values)
                self.calculated=True
            except Exception:
                return False
        return True
    
    def print(self):
        print(str(self))
    
    def __str__(self):
        if self.calc():
            return ", ".join([str(v) for v in self.values])
        return ""
    
    def getDataFrame(self):
        if self.calc():
            y_real=pd.DataFrame(self.y_real, columns=[ "y_real_%d" % i for i in range(1, 1+self.y_real.size(1))])
            y_pred=pd.DataFrame(self.y_pred, columns=[ "y_pred_%d" % i for i in range(1, 1+self.y_real.size(1))])
            df = pd.concat((y_real, y_pred), axis=1)
            if hasattr(self, "names"):
                df.index = self.names
            return df
        else:
            return None
    
    def getValuesSeries(self, name="index"):
        values = [v.value for v in self.values]
        index = [v.name for v in self.values]
        return pd.Series(values, index = index, name = name)
    
    @abstractmethod
    def _calc_index(self, y_pred, y_real):
        pass
    
    def __repr__(self):
        return str(self)

class CatogoryIndex(IndexBase):
    def _calc_index(self, y_pred, y_real):
        
        y_pred_cat=torch.argmax(y_pred, dim=1)
        y_real_cat=torch.argmax(y_real, dim=1)
        self.y_pred_sm = torch.softmax(y_pred, dim = 1)
        
        if(len(y_real_cat) > 0):
            ACC=accuracy_score(y_real_cat, y_pred_cat)
        return [
            IndexValue("ACC", ACC, "%.3f")
        ]

class BinaryCatogoryIndex(CatogoryIndex):
    def _calc_index(self, y_pred, y_real):
        
        y_pred_cat=torch.argmax(y_pred, dim=1)
        y_real_cat=torch.argmax(y_real, dim=1)
        self.y_pred_sm = torch.softmax(y_pred, dim = 1)
        
        if(len(y_real_cat) > 0):
            ACC=accuracy_score(y_real_cat, y_pred_cat)
            try:
                AUC=roc_auc_score(y_real_cat, y_score=self.y_pred_sm[:, 1])
            except ValueError:
                AUC=0.0
        else:
            ACC=0
            AUC=0
        TP=0
        TN=0
        FP=0
        FN=0
        for p, r in zip(y_pred_cat, y_real_cat):
            if(p==1):
                if(r==1):
                    TP+=1
                else:
                    FP+=1
            else:
                if(r==1):
                    FN+=1
                else:
                    TN+=1
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        MCC = (TP*TN - FP*FN) / sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) if (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) > 0 else 0
        
        return [
            IndexValue("AUC", AUC, "%.3f"), 
            IndexValue("ACC", ACC, "%.3f"), 
            IndexValue("precision", precision, "%.3f"), 
            IndexValue("recall", recall, "%.3f"), 
            IndexValue("F1", F1, "%.3f"), 
            IndexValue("MCC", MCC, "%.3f"), 
            IndexValue("TP", TP), 
            IndexValue("FP", FP), 
            IndexValue("TN", TN), 
            IndexValue("FN", FN), 
        ]

class SingleValueIndex(IndexBase):
    def _calc_index(self, y_pred, y_real):
        if(len(y_pred) > 0):
            RMSE=torch.nn.functional.mse_loss(y_pred, y_real).sqrt().item()
            pearson = pearsonr(y_pred.numpy().reshape(-1), y_real.numpy().reshape(-1)).statistic
        else:
            RMSE=float("nan")
            pearson = 0
        
        return [
            IndexValue("RMSE", RMSE, "%.3f"), 
            IndexValue("R", pearson, "%.3f")
        ]

class SingleValueCategoryIndex(SingleValueIndex):
    def _calc_index(self, y_pred, y_real):
        if(len(y_pred) > 0):
            RMSE=torch.nn.functional.mse_loss(y_pred, y_real).sqrt().item()
            pearson = pearsonr(y_pred.numpy().reshape(-1), y_real.numpy().reshape(-1)).statistic
        else:
            RMSE=float("nan")
            pearson = 0
        
        y_pred_cat=torch.zeros(y_pred.size(0))
        y_pred_cat[y_pred[:, 0] > 0] = 1
        y_real_cat=torch.zeros(y_real.size(0))
        y_real_cat[y_real[:, 0] > 0] = 1
        # self.y_pred_sm = torch.sigmoid(y_pred, dim = 1)
        
        if(len(y_real_cat) > 0):
            ACC=accuracy_score(y_real_cat, y_pred_cat)
            try:
                AUC=roc_auc_score(y_real_cat, y_score=y_pred[:, 0])
            except ValueError:
                AUC=0.0
        else:
            ACC=0
            AUC=0
        TP=0
        TN=0
        FP=0
        FN=0
        for p, r in zip(y_pred_cat, y_real_cat):
            if(p==1):
                if(r==1):
                    TP+=1
                else:
                    FP+=1
            else:
                if(r==1):
                    FN+=1
                else:
                    TN+=1
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        MCC = (TP*TN - FP*FN) / sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) if (TP + FP)*(TP + FN)*(TN + FP)*(TN + FN) > 0 else 0
        
        return [
            IndexValue("RMSE", RMSE, "%.3f"), 
            IndexValue("R", pearson, "%.3f"),
            IndexValue("AUC", AUC, "%.3f"), 
            IndexValue("ACC", ACC, "%.3f"), 
            IndexValue("precision", precision, "%.3f"), 
            IndexValue("recall", recall, "%.3f"), 
            IndexValue("F1", F1, "%.3f"), 
            IndexValue("MCC", MCC, "%.3f"), 
            IndexValue("TP", TP), 
            IndexValue("FP", FP), 
            IndexValue("TN", TN), 
            IndexValue("FN", FN), 
        ]



