from kipoiseq.transforms.functional import one_hot
from pyfaidx import Fasta
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from scipy import stats
import torch.optim as optim
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler
import h5py
import random
import os

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子

setup_seed(1)

class scDataset(Dataset):
    def __init__(self,onehotdict,beddf,idx,col_name):
        
        #get train y value 
        self.y_value=np.array(beddf.loc[idx,col_name])
        print(self.y_value.shape)

        self.promotername=idx
        print(len(self.promotername))

        # get corresponding train X value 
        onehotls=[onehotdict[promoter_id] for promoter_id in idx]
        
        #print(len(onehotls))
        for promoter_id in idx:
          if len(onehotdict[promoter_id])!=20000:
            print(promoter_id)
          #print(len(onehotdict[promoter_id]))
        
        
        
        # onehotls=[]
        # for i in self.gn:
        #     print(i)
        #     onehotls.append(onehotdict[i])
        #print(onehotls)
            
        self.combine_array=np.stack(onehotls)
        # print(self.combine_array)
                
    def __len__(self):
        return len(self.y_value)
    
    
    def __getitem__(self,index):
        #print(self.combine_array.shape)
        #print(index)
        seq=np.transpose(self.combine_array[index]).astype(np.float32)
        y_value=self.y_value[index].astype(np.float32)
        TSS_name=self.promotername[index]
        
        #print(TSS_name)

        sample = {"seq": seq, "y_true_value": y_value,'TSS_name':TSS_name}
        #print(sample)
        return sample 


        

# Validation

def evaluate(device,net,dataloader,criterion):
    net.eval()
    avg_loss=0
    y_true_ls=[]
    y_pred_ls=[]
    TSS_name_ls=[]
    
    with torch.no_grad():
        for data in dataloader:
            seq=data['seq'].to(device,dtype=torch.float)
            #hl=data['hl'].to(device,dtype=torch.float)
            #print(seq)
            y_true_value=data['y_true_value'].to(device,dtype=torch.float)
            
            TSS_name=data['TSS_name']
        
            y_pred_value=net(seq)
            current_loss=criterion(y_true_value.squeeze(),y_pred_value.squeeze())
            avg_loss+=current_loss.item() / len(dataloader)
            # print("True y:")
            # print(y_true_value)
            
            # print("Pred y:")
            # print(y_pred_value)
            #print(TSS_name)
            
            y_true_ls.append(y_true_value.cpu().numpy().squeeze())
            y_pred_ls.append(y_pred_value.cpu().numpy().squeeze())
            TSS_name_ls.append(TSS_name[0])
            
        # print("True ")
        # print(y_true_ls)
    return avg_loss,y_true_ls,y_pred_ls,TSS_name_ls


    

import matplotlib.pyplot as plt

# Train

def train(device,train_loader,validation_loader,net,criterion,learning_rate,lr_sched,num_epochs,logs_dir,model_dir):

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)




    
    logger = SummaryWriter(logs_dir)
    optimizer=optim.Adam(net.parameters(),lr=learning_rate)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.1, patience=5)
    best_loss=100
    
    for epoch in range(0,num_epochs):
        net.train()
        avg_train_loss=0
        for data in train_loader:
            seq=data['seq'].to(device,dtype=torch.float)
            # print("Training seq:")
            # print(seq)
            #hl=data['hl'].to(device,dtype=torch.float)
            y_true_value=data['y_true_value'].to(device,dtype=torch.float)
            #print(y_true_value.shape)
            
            # plt.hist(data['y_true_value'],bins=100)
            # plt.title('the distribution of true y value')
            # plt.show()
            
            
            

            
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                
                #Forward pass
                y_pred_value=net(seq)
                # print('Training y prediction')
                # print(y_pred_value.shape)
                
                # print("this is prediction one  dimension")
                # print(y_pred_value.squeeze().shape)
                
                # print("this is True one  dimension")
                # print(y_true_value.squeeze().shape)

                current_loss = criterion(y_pred_value.squeeze(), y_true_value.squeeze())  # why it does not change?
                current_loss.backward()
                optimizer.step()
                avg_train_loss+=current_loss.item()/len(train_loader)
                logger.add_scalar('train_loss_epoch', avg_train_loss, epoch + 1)
                
                
        print("The average train loss is %f" %avg_train_loss)
        
        
        #get validation loss 
        validation_loss,y_true_ls,y_pred_ls,TSS_name_ls=evaluate(device,net,validation_loader,criterion)

        print('Validation correlation:')
        corr_value=stats.pearsonr(y_true_ls, y_pred_ls)
        print(corr_value)
        
        print("The average validation loss is %f" %validation_loss)

        logger.add_scalar('valid_loss_epoch', validation_loss, epoch + 1)
        
        
        # store best model
        if validation_loss<best_loss:
            best_loss=validation_loss
            state={'epoch':epoch+1,'state_dict':net.state_dict(),'optimizer':optimizer.state_dict(),'loss':criterion,'scheduler':scheduler.state_dict()}
            torch.save(state,model_dir+'/model_last.pth')
            
            
        # LR scheduler on plateau (based on validation loss)
        if lr_sched:
            scheduler.step(validation_loss)
            
        print("Finish training.")
        

        

        

            

# Test

def test(testdataloader,device,net,model_dir,save_file,fold_no):
    
    #get model
    check_point=torch.load(model_dir+'/model_last.pth',weights_only=False)
    net.load_state_dict(check_point['state_dict'])
    criterion = check_point['loss']
    
    test_loss,y_true_value,y_pred_value,TSS_name_value=evaluate(device,net,testdataloader,criterion)

    #y_true_value_flatten=[j for i in y_true_value for j in i]
    #y_pred_value_flatten=[j for i in y_pred_value for j in i]


    

    
    #y_true_value=np.array([i[0]for i in y_true_value])
    corr_value=stats.pearsonr(y_true_value, y_pred_value)
    # print("True y value: \n")
    # print(y_true_value)
    
    # print("Pred y value: \n")
    # print(y_pred_value)
    
    # Save predictions
    if save_file is not None:
        pickle.dump({'y_true': y_true_value, 'y_pred': y_pred_value,'TSS':TSS_name_value}, open(save_file, 'wb'))
    
    print("Test loss is %f"%test_loss)
    print(corr_value)
    #print("The pearson's correlation is %f" %corr_value)

    

# Model

class CNN1D(nn.Module):
    def __init__(self,):
        super(CNN1D, self).__init__()
                
        # Convolutional layers
        conv_layers = []        
        conv_layers.append(nn.Conv1d(4, 128, kernel_size = 6, padding = 2))
        conv_layers.append(nn.LeakyReLU())
        conv_layers.append(nn.MaxPool1d(30, padding = 1))
        conv_layers.append(nn.Conv1d(128, 32, kernel_size = 9, padding = 4))
        conv_layers.append(nn.LeakyReLU())
        conv_layers.append(nn.MaxPool1d(10))        
        conv_layers.append(nn.Flatten())
        
        self.conv = nn.Sequential(*conv_layers)
        
        #Fully connected layers
        fc_layers = []
        fc_layers.append(nn.Linear(2112, 64))
        fc_layers.append(nn.LeakyReLU())
        fc_layers.append(nn.Dropout(p=0.00099))
        fc_layers.append(nn.Linear(64, 2))
        fc_layers.append(nn.LeakyReLU())
        fc_layers.append(nn.Dropout(p=0.01546))
        fc_layers.append(nn.Linear(2,1))

        self.fc = nn.Sequential(*fc_layers)

                
    def forward(self, seq):           
        
        x1 = self.conv(seq)
        # print(x1.shape)
        #x2 = hl
        #x = torch.cat((x1,x2), dim=1)
        x = self.fc(x1)
        
        return x


    

# Main

#device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device=2

device

with open('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/Human/protein_coding_promoter_onehot.pkl','rb') as f:
    onehotdict=pickle.load(f)
    

allyvaluedf=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/Human/protein_coding_promoter.csv',index_col=0)
allyvaluedf.set_index('promoterId',inplace=True,drop=True)
allyvaluedf



mergedf=allyvaluedf.copy()
mergedf




organtimepointls=['Human_Kidney_0dpb.mean']
organtimepointls

kf=KFold(n_splits=5,random_state=666,shuffle=True)
kf

for fold_no, (train_index, test_index) in enumerate(kf.split(mergedf)):
    print("this is fold number: %i" %fold_no)


    
    # print(train_index)
    # print(test_index)
    
    

    np.random.seed(666)
    
    idx_rest = np.random.permutation(len(mergedf)-len(test_index))
    
    idx_val = train_index[idx_rest[:int(0.2*len(idx_rest))]]
    idx_train = train_index[idx_rest[int(0.2*len(idx_rest)):]]
    
    print("train_id")
    print(idx_train[:10])
    
    print("validation_id")
    print(idx_val[:10])
    



    train_promoterID=[mergedf.index.tolist()[i] for i in idx_train]
    test_promoterID=[mergedf.index.tolist()[i] for i in test_index]
    validation_promoterID=[mergedf.index.tolist()[i] for i in idx_val]
    


    

    


    train_set=scDataset(onehotdict,mergedf,train_promoterID,organtimepointls)
    validation_set=scDataset(onehotdict,mergedf,validation_promoterID,organtimepointls)
    test_set=scDataset(onehotdict,mergedf,test_promoterID,organtimepointls)

    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


    

    model_dir='/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/Human/model_dir'
    log_dir='/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/Human/log_dir'
    save_dir='/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/Human/save_dir/TSS_true_pred_y_'+str(fold_no)+'.pkl'


    criterion = torch.nn.MSELoss().to(device)
    net = CNN1D().to(device)
    init_lr =0.0005
    lr_sched = True
    num_epoch=50
    
    train(device,train_loader,val_loader,net,criterion,init_lr,lr_sched,num_epoch,log_dir,model_dir)
    test(test_loader,device,net,model_dir,save_dir,fold_no)

    

