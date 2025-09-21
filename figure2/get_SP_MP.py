import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from pymccorrelation import pymccorrelation
import os
from hilearn.plot import corr_plot


allfile=os.listdir('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/Mouse')
allfile

mouse_activity=[i for i in allfile if ('activity' in i)&('Liver' in i) ]
mouse_activity

allfilels=[]
for i in mouse_activity:
    inputfilePath='/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/Mouse/'+i
    df=pd.read_csv(inputfilePath,delimiter='\t')
    allfilels.append(df)
    print(df)

all_liverdf=pd.concat(allfilels,axis=1)
all_liverdf

rowdf=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/Mouse/Mouse_Liver_10.5_rowData.tsv',delimiter='\t')
rowdf

rowdf.drop(labels=['txId','Mouse_Liver_10.5.mean','Mouse_Liver_10.5.gene.mean','Mouse_Liver_10.5.class'],axis=1,inplace=True)
rowdf

rowdf['promoter_info']=rowdf['promoterId'].astype('str')+'_'+rowdf['geneId']+'_'+rowdf['seqnames']+'_'+rowdf['start'].astype('str')+'_'+rowdf['internalPromoter'].astype('str')+'_'+rowdf['promoterPosition'].astype('str')
rowdf

rowdf=rowdf[['promoter_info']]
rowdf

alldf=pd.concat([rowdf,all_liverdf],axis=1)
alldf

alldf.set_index('promoter_info',inplace=True)
alldf

keepdf=alldf[~(alldf==0).all(axis=1)]
keepdf

keepdf.reset_index(inplace=True)
keepdf

meltdf=keepdf.melt(id_vars='promoter_info')
meltdf

meltdf['time']=meltdf['variable'].str.split('.',expand=True)[3]
meltdf

meltdf['time']=np.where(meltdf['time'].str.contains('pb'),meltdf['time'],meltdf['time']+'.5')
meltdf

meltdf['gene_id']=meltdf['promoter_info'].str.split('_',expand=True)[1]
meltdf

genesumdf=pd.DataFrame(meltdf.groupby(['time','promoter_info'])['value'].sum())
genesumdf

genesumdf.reset_index(inplace=True)
genesumdf

genesumkeepdf=genesumdf[genesumdf['value']!=0]
genesumkeepdf

genesumkeepdf['gene_id']=genesumkeepdf['promoter_info'].str.split('_',expand=True)[1]
genesumkeepdf

onetimedf=genesumkeepdf[genesumkeepdf['time']=='0dpb']
onetimedf

timepoint=set(genesumkeepdf['time'])
timepoint

alltimels=[]
for i in timepoint:
    onetimedf=genesumkeepdf[genesumkeepdf['time']==i]
    duplicatedgene=onetimedf[onetimedf.duplicated('gene_id',keep=False)]
    duplicatedgene['class']='MP'
    nonduplicatedgene=onetimedf[~onetimedf.duplicated('gene_id',keep=False)]
    nonduplicatedgene['class']='SP'
    allgenedf=pd.concat([nonduplicatedgene,duplicatedgene],axis=0)
    allgenedf=allgenedf[['time','gene_id','class']]
    allgenedf.drop_duplicates('gene_id',inplace=True)
    alltimels.append(allgenedf)
    




alltimels

finaldf=pd.concat(alltimels,axis=0)
finaldf

finaldf.to_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/SP_MP_gene/SP_MP_each_timepoints.csv')

