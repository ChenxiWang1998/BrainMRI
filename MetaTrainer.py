import math

import torch

import MetaBrain
import MetaData

class OutputSubTrainer:
    def __init__(self, name, loss_function, index_type):
        self.name=name
        self.loss_function=loss_function
        self.index_type=index_type
    
    def generateIndex(self):
        return self.index_type()

class MetaTrainer:
    def __init__(self, model:MetaBrain.MetaBrainViT, subtrainers:list[OutputSubTrainer] = None, opt = None):
        self.model=model
        self.trainable=False
        if subtrainers is not None:
            self.set_subtrainers(subtrainers = subtrainers)
            self.opt = opt if opt is not None else torch.optim.Adam(model.parameters())
    
    def set_subtrainers(self, subtrainers):
        self.subtrainers=subtrainers
        self.__check_names()
        self.trainable = True
    
    def __check_names(self):
        model_names=list(self.model.output_modules.keys())
        for st in self.subtrainers:
            name=st.name
            assert name in model_names, "illegal subtrainer:%s" % name
            model_names.remove(name)
        assert len(model_names) == 0, "need output trainer:%s" % model_names
    
    def evalLoader(self, loader):
        pred_list_dict = dict([[s, list()] for s in self.model.output_modules.keys()])
        with torch.no_grad():
            for x, x_mask, _, _ in loader:
                pred = self.model(x, x_mask)
                for target_name, y_pred_sub in pred.items():
                    pred_list_dict[target_name].append(y_pred_sub.cpu())
        for target_name in pred_list_dict.keys():
            pred_list = pred_list_dict[target_name]
            pred_list_dict[target_name] = torch.concat(pred_list, dim = 0)
        return pred_list_dict
    
    def calcLoader(self, loader:MetaData.MetaDataLoader, train=False, verbose = False):
        if train and not self.trainable:
            print("trainer is not trainable")
        train = train and self.trainable
        self.model.train(train)
        if not train:
            prev = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
        
        index_dict=dict([[trainer.name, trainer.generateIndex()] for trainer in self.subtrainers]) #新建每个target指标
        loader.setDevice(self.model.device)
        it=0
        for names, x, x_mask, y_real, y_mask in loader:
            it+=1
            if(verbose):
                print("iter:", it)
            y_pred=self.model(x, x_mask)
            loss = 0
            for trainer in self.subtrainers:
                name=trainer.name
                y_mask_sub=y_mask[name]
                if not y_mask_sub.all():
                    y_pred_masked=y_pred[name][torch.logical_not(y_mask_sub)] #提取未mask的数据，在mask中被掩盖的位置为True，因此需将其转换为False
                    y_real_masked=y_real[name][torch.logical_not(y_mask_sub)]
                    names_masked = [n for n, m in zip(names, y_mask_sub) if m == False]
                    index_dict[name].record(y_pred=y_pred_masked, y_real=y_real_masked, names = names_masked)
                    l=trainer.loss_function(y_pred_masked, y_real_masked)
                    loss+=l
                if(verbose):
                    print(trainer.name, l.item(), len(y_pred_masked))
            if(train):
                if(verbose):
                    if math.isnan(loss.item()):
                        return x, x_mask, y_real, y_mask
                    else:
                        torch.save(self.model, "model_cache.pth")
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        
        if not train:
            torch.set_grad_enabled(prev)
        
        return index_dict

