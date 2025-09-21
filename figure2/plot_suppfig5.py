import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from pymccorrelation import pymccorrelation
import os

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/alternative_test_with_earliest/up_down_No.csv',index_col=0)
df

dfT=df.T
dfT

dfT.reset_index(inplace=True)
dfT

dfT['organ']=dfT['index'].str.split('_',expand=True)[1]
dfT

dfT['time']=dfT['index'].str.split('_',expand=True)[2].str.split('.rds',expand=True)[0]
dfT

df=dfT.copy()
df

def plot_no(organ):
    organdf=df[df['organ']==organ]
    organdf=organdf[[1,2,'time']]
    organdf.columns=['Downregulated','Upregulated','time']
    meltdf=organdf.melt(id_vars='time')
    custome_order=['11.5','12.5','13.5','14.5','15.5','16.5','17.5','18.5','0dpb','3dpb','2wpb','4wpb','9wpb']
    meltdf['time']=pd.Categorical(meltdf['time'],ordered=True,categories=custome_order)
    meltdf.sort_values('time',inplace=True)
    namedict={}
    for i in custome_order:
        namedict[i]=i+' '+'VS 10.5'
    print(namedict)
        
    

    # namedict={'11.5':'10.5 vs 1.5','12.5':'12.5 vs 13.5','13.5':'13.5 vs 14.5',
    #          '14.5':'14.5 vs 15.5','15.5':'15.5 vs 16.5','16.5':'16.5 vs 17.5','17.5':'17.5 vs 18.5',
    #           '18.5':'18.5 vs 0dpb','0dpb':'0dpb vs 3dpb',
    #          '3dpb':'3dpb vs 2wpb','2wpb':'2wpb vs 4wpb','4wpb':'4wpb vs 9wpb'}
    meltdf['label']=meltdf['time'].map(namedict)
    plt.figure(figsize=(7,5))
    sns.barplot(data=meltdf,x='label',y='value',hue='variable',palette=["#5861AC",'#F28080'],width=0.9,dodge=True,edgecolor='black', linewidth=1)
    plt.ylabel("No. of Gene",fontsize=14)
    plt.xticks(fontsize=12,rotation=45)
    plt.xlabel("")
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(title=None)
    plt.title(organ,fontsize=15)
    outputpath=os.path.join('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/num_change',organ+'.pdf')
    
    plt.savefig(outputpath,dpi=300,bbox_inches='tight')
    plt.show()

organls=['Ovary','Cerebellum','Brain','Liver','Heart','Testis','Kidney']

for i in organls:
    plot_no(i)
