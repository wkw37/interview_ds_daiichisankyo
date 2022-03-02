import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pickle
# for simple ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
# for NN
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from datetime import datetime
# for metrics
from sklearn.inspection import permutation_importance
from sklearn.metrics import *

class Data_Loader:
    cols_X = None
    y_label = None
    X,y = None, None
    cont_cols,cat_cols = [],[]
    def __init__(self,file_path,train=False,cont_cols=['duration','age'], dropna=False):
        """
        Input
        ------------------------
        file_path : str
            Input file path (.xlsx)
        train : bool
            If the data is a training data
        cont_cols : list
            List of column names containing continuous values (all others will be forced into strings)
        """
        self.file_path = file_path
        self.data = pd.read_excel(self.file_path,engine='openpyxl')
        if dropna: self.data = self.data[self.data.apply(lambda x: 'unknown' not in x.values.tolist(),axis=1)]
        cols = self.data.columns.tolist()
        
        if train:
            # Training set
            self.y_label = 'y'
            assert self.y_label in cols
            self.cols_X = sorted(list(set(cols)-set([self.y_label])))
            self.y = self.data[self.y_label]
            
        else: 
            # Test set
            self.cols_X = sorted(cols)

        self.cont_cols = cont_cols
        self.cat_cols = sorted(set(self.cols_X)-set(cont_cols))
        for cat_col in self.cat_cols:
            self.data[cat_col] = self.data[cat_col].astype("category")
            
        self.X = self.data[self.cols_X]
            
class EDA:
    data = None
    def __init__(self,data_loader_obj):
        self.data_loader_obj = data_loader_obj
        self.run()
    def run(self):
        # get summary
        print("Continuous variables")
        print(self.data_loader_obj.data[self.data_loader_obj.cont_cols].describe())
        # Plot distributions
        for col in self.data_loader_obj.cont_cols:
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            self.data_loader_obj.data[col].hist(ax=ax)
            fig.suptitle(col)
        
        # find most common
        print("The observations in each categorical variable sorted by frequency")
        for col in self.data_loader_obj.cat_cols:
            print(col,self.data_loader_obj.data[col].value_counts().to_dict().items())
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            self.data_loader_obj.data[col].value_counts().plot.bar(ax=ax)
            fig.suptitle(col)
        
        # outcome-related observations
        for col in self.data_loader_obj.cont_cols:
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            sns.violinplot(data=self.data_loader_obj.data, x='y',y=col,ax=ax)
            fig.suptitle(col)
        for col in self.data_loader_obj.cat_cols:
            fig,ax = plt.subplots(1,1,figsize=(5,5))
            self.data_loader_obj.data.groupby('y').apply(lambda x: x[col].value_counts()).plot.bar(ax=ax)
            fig.suptitle(col)
            
class NNDataLoader(Dataset):
    def __init__(self, X, Y, embedded_col_names):
        X = X.copy()
        self.X1 = X.loc[:,embedded_col_names].copy().values.astype(np.int64) #categorical columns
        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32) #numerical columns
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]
class Basic_ML_Classifier:
    def __init__(self,data_loader_obj):
        self.data_loader_obj = data_loader_obj
        self.X = self.data_loader_obj.X
        self.cont_cols = self.data_loader_obj.cont_cols
        self.cat_cols = self.data_loader_obj.cat_cols# encoding
        self.y = LabelEncoder().fit_transform(self.data_loader_obj.y)
        self.y_preds = {}
        
RF = RandomForestClassifier(max_depth=1000, random_state=0)
GB = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,
                                max_depth=2, random_state=0)
Ridge = RidgeClassifier()
class All_Simple(Basic_ML_Classifier):
    def __init__(self,data_loader_obj):
        super().__init__(data_loader_obj)
        new_X = [self.X.drop(columns=self.cat_cols)]
        for col in self.cat_cols:
            _df = pd.get_dummies(self.X[col])
            _df.rename(columns = {x:f'{col}_{x}' for x in _df.columns},inplace=True)
            new_X.append(_df)
        self.X = pd.concat(new_X,axis=1).values
        self.run()
    def run(self):
        kf = KFold(n_splits=10)
        for clf_name,clf in zip(['RF','GB','Ridge'],[RF,GB,Ridge]):
            self.y_preds.update({clf_name:{'y_valid':[],'y_pred':[]}})
            for train_index, valid_index in kf.split(self.X):
                print("TRAIN:", train_index, "VALID:", valid_index)
                X_train, X_valid = self.X[train_index], self.X[valid_index]
                y_train, y_valid = self.y[train_index], self.y[valid_index]
        
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_valid)
                self.y_preds[clf_name]['y_valid'] += list(y_valid)
                self.y_preds[clf_name]['y_pred'] += list(y_pred)
class SimplePred_Ridge:
    def __init__(self,data_loader_obj_train,data_loader_obj_test):
        self.data_loader_obj_train = data_loader_obj_train
        self.data_loader_obj_test = data_loader_obj_test
        
        # X
        self.X_train = self.data_loader_obj_train.X
        self.X_test = self.data_loader_obj_test.X
        self.cont_cols = self.data_loader_obj_train.cont_cols
        self.cat_cols = self.data_loader_obj_train.cat_cols# encoding
        
        # Y
        self.y_train = LabelEncoder().fit_transform(self.data_loader_obj_train.y)
        
        # label X
        new_X = [self.X_train.drop(columns=self.cat_cols)]
        for col in self.cat_cols:
            _df = pd.get_dummies(self.X_train[col])
            _df.rename(columns = {x:f'{col}_{x}' for x in _df.columns},inplace=True)
            new_X.append(_df)
        self.X_train = pd.concat(new_X,axis=1)
        self.new_X_cols = self.X_train.columns.tolist()
        self.X_train = self.X_train.values
        
        new_X = [self.X_test.drop(columns=self.cat_cols)]
        for col in self.cat_cols:
            _df = pd.get_dummies(self.X_test[col])
            _df.rename(columns = {x:f'{col}_{x}' for x in _df.columns},inplace=True)
            new_X.append(_df)
        self.X_test = pd.concat(new_X,axis=1)
        self.X_test = self.X_test.reindex(columns=self.new_X_cols).fillna(0.0)[self.new_X_cols].values
        self.y_pred = None
        self.run()
    def run(self):
        # train and run on all
        clf = Ridge
        clf.fit(self.X_train, self.y_train)
        self.y_pred = clf.predict(self.X_test)
    
# Neural network
# Based on https://jovian.ai/aakanksha-ns/shelter-outcome
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class NNModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont
        print("self.n_emb",self.n_emb)
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 5)
        #self.lin3 = nn.Linear(50, 5)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        #self.bn3 = nn.BatchNorm1d(50)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)
        

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = self.lin2(x)
        #x = F.relu(self.lin2(x))
        #x = self.drops(x)
        #x = self.bn3(x)
        #x = self.lin3(x)
        return x
class NNDataLoader(Dataset):
    def __init__(self, X1,X2,y):
        self.X1 = X1
        self.X2 = X2
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]

class NN(Basic_ML_Classifier):
    def __init__(self,data_loader_obj):
        super().__init__(data_loader_obj)
        # embedding
        embedded_cols = {n: len(col.cat.categories) for n,col in self.X.items() if n in self.cat_cols}
        self.embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]
        # Label Encode categorical columns in X
        for col in self.cat_cols:
            self.X[col] = LabelEncoder().fit_transform(self.X[col])
        self.X1 = self.X.loc[:,self.cat_cols].copy().values.astype(np.int64)
        self.X2 = self.X.drop(columns=self.cat_cols).copy().values.astype(np.float32)
        self.y = torch.LongTensor(self.y)
        self.y_preds.update({'NN':{'y_valid':[],'y_pred':[]}})
        #self.run()
    def run(self):
        device = get_default_device()
        
        # history
        self.history = {'loss':[],'val_loss':[]}
        batch_size = 1000
        kf = KFold(n_splits=10)
        # model selection
        for train_index, valid_index in kf.split(self.X1):
            print("TRAIN:", train_index, "VALID:", valid_index)
            X1_train, X1_valid = self.X1[train_index], self.X1[valid_index]
            X2_train, X2_valid = self.X2[train_index], self.X2[valid_index]
            y_train, y_valid = self.y[train_index], self.y[valid_index]
        
            train_ds = NNDataLoader(X1_train,X2_train,y_train)
            valid_ds = NNDataLoader(X1_valid,X2_valid,y_valid)
            
            model = NNModel(self.embedding_sizes, len(self.cont_cols))
            to_device(model, device)
            train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
            valid_dl = DataLoader(valid_ds, batch_size=batch_size,shuffle=True)
            train_dl = DeviceDataLoader(train_dl, device)
            valid_dl = DeviceDataLoader(valid_dl, device)
            self.train_loop(model, train_dl=train_dl, valid_dl=valid_dl, epochs=8, lr=0.05, wd=0.00001)
        
    def train_model(self,model, optim, train_dl):
        model.train()
        total = 0
        sum_loss = 0
        for x1, x2, y in train_dl:
            batch = y.shape[0]
            output = model(x1, x2)
            loss = F.cross_entropy(output, y)   
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += batch*(loss.item())
        return sum_loss/total
    def val_loss(self,model, valid_dl):
        model.eval()
        total = 0
        sum_loss = 0
        correct = 0
        for x1, x2, y in valid_dl:
            current_batch_size = y.shape[0]
            out = model(x1, x2)
            loss = F.cross_entropy(out, y)
            sum_loss += current_batch_size*(loss.item())
            total += current_batch_size
            pred = torch.max(out, 1)[1]
            correct += (pred == y).float().sum().item()
            self.y_preds['NN']['y_valid'].append(y.detach().numpy())
            self.y_preds['NN']['y_pred'].append(pred.detach().numpy())

        self.history['val_loss'].append(sum_loss/total)
        print("valid loss %.3f and accuracy %.3f" % (sum_loss/total, correct/total))
        return sum_loss/total, correct/total
    def get_optimizer(self,model, lr = 0.001, wd = 0.0):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
        return optim
    def train_loop(self,model, train_dl,valid_dl,epochs, lr=0.01, wd=0.0):
        optim = self.get_optimizer(model, lr = lr, wd = wd)
        for i in range(epochs): 
            loss = self.train_model(model, optim, train_dl)
            self.history['loss'].append(loss)
            print("training loss: ", loss)
            self.val_loss(model, valid_dl)

# Feature importance
class Feat_Imp(Basic_ML_Classifier):
    def __init__(self,data_loader_obj):
        super().__init__(data_loader_obj)
        new_X = [self.X.drop(columns=self.cat_cols)]
        for col in self.cat_cols:
            _df = pd.get_dummies(self.X[col])
            _df.rename(columns = {x:f'{col}_{x}' for x in _df.columns},inplace=True)
            new_X.append(_df)
        new_X = pd.concat(new_X,axis=1)
        self.X_new_cols = new_X.columns
        self.X = new_X.values
        self.feat_imp = []
        self.run()
    def run(self):
        clf = RF
        kf = KFold(n_splits=10)
        for train_index, valid_index in kf.split(self.X):
            print("TRAIN:", train_index, "VALID:", valid_index)
            X_train, X_valid = self.X[train_index], self.X[valid_index]
            y_train, y_valid = self.y[train_index], self.y[valid_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_valid)
            result = permutation_importance(
                clf, X_valid, y_valid, n_repeats=10, random_state=42
            )
            print(result)
            #sorted_idx = result.importances_mean.argsort()
            #self.feat_imp.append(result.importances[sorted_idx])
            self.feat_imp.append(result)
## Metrics
class Evaluate:
    def __init__(self):
        self.metrics =[accuracy_score,f1_score,matthews_corrcoef,precision_score,recall_score]
        with open('drop_na/preds_simple.pickle','rb') as f:
            preds = pickle.load(f)
        with open('drop_na/preds_nn.pickle','rb') as f:
            preds_nn = pickle.load(f)
        preds.update(preds_nn)
        self.preds = preds.copy()
        self.performance = {}
        self.calculate()
    def calculate(self):
        
        for method in self.preds:
            y_true = self.preds[method]['y_valid']
            y_pred = self.preds[method]['y_pred']
            if method=='NN':
                y_true = list(itertools.chain.from_iterable(y_true))
                y_pred = list(itertools.chain.from_iterable(y_pred))
            self.performance.update({method:{metric.__name__:metric(y_true,y_pred) for metric in self.metrics}})
            