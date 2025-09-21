import pandas as pd
import os

allfile=os.listdir('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/Mouse')
allfile_exp=[i for i in allfile if '_exp' in i]
allfile_exp




countdict={}
for i in allfile_exp:
    print(i)
    inputfile=os.path.join('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/Mouse/',i)
    df=pd.read_csv(inputfile,index_col=0)
    promoterinput=os.path.join('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/Mouse',i.split('_exp.csv')[0]+'_rowData.tsv')
    promoterdf=pd.read_csv(promoterinput,delimiter='\t')
    activatedf=promoterdf[promoterdf[i.split('_exp')[0]+'.class'].isin(['Major','Minor'])]
    activatenum=len(activatedf)
    print(activatenum)

    expdf=pd.concat([df,promoterdf],axis=1)
    expdf.set_index('geneId',inplace=True)
    expdf.drop(labels=['promoterId','seqnames','start','strand','internalPromoter','promoterPosition','txId',i.split('_exp.csv')[0]+'.class'],axis=1,inplace=True)
    dfT=expdf.T
    totalgene=pd.DataFrame(dfT.sum(axis=0))
    totalgene.reset_index(inplace=True)
    keepdf=totalgene.drop_duplicates('geneId')
    expnum=len(keepdf[keepdf[0]!=0])
    print(expnum)

    countdict[i]=activatenum/expnum

    
    


countdict







numdf=pd.DataFrame(countdict,index=[0])
numdf

numdfT=numdf.T
numdfT

numdfT.reset_index(inplace=True)
numdfT

numdfT['time']=numdfT['index'].str.split('_',expand=True)[2]
numdfT

numdfT['organ']=numdfT['index'].str.split('_',expand=True)[1]
numdfT

custome_order=['10.5','11.5','12.5','13.5','14.5','15.5','16.5','17.5','18.5','0dpb','3dpb','2wpb','4wpb','9wpb']


numdfT['time']=pd.Categorical(numdfT['time'],ordered=True,categories=custome_order)
numdfT

numdfT.sort_values('time',inplace=True)
numdfT



import seaborn as sns
import matplotlib.pyplot as plt

lut={'10.5':'#e64b35cc','11.5':'#4dbbd5cc','12.5':'#00a087cc','13.5':'#9fdaf7',
    '14.5':'#f39b7fcc','15.5':'#8491b4cc','16.5':'#91d1c2cc','17.5':'#57919d','18.5':'#7e6148cc',
    '0dpb':'#d9dee7','3dpb':'#ecb426','2wpb':'#416594','4wpb':'#f9b3ad','9wpb':'#c9a1ca'}

fig=plt.figure(figsize=(5,4))
sns.boxplot(data=numdfT,x='time',y=0,palette=lut,width=0.8)
plt.xticks(rotation=60)
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
plt.xlabel("")
plt.ylabel("Normalied No. of activate promoters",fontsize=13)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/activate_promoter_num_timepoint.pdf',dpi=300,bbox_inches='tight')

from scipy.stats import f_oneway


groups = [group[0].values for name, group in numdfT.groupby('time')]


groups

f_stat, p_value = f_oneway(*groups)


f_stat

p_value

organdict={'Ovary':"#CC3399",'Liver':"#339900",'Heart':"#CC0000",'Brain':"#3399CC",'Testis':"#FF6600",
            'Kidney':"#CC9900",'Cerebellum':"#33CCFF"}

organ_order=['Brain','Cerebellum','Heart','Kidney','Liver','Ovary','Testis']

numdfT['organ']=pd.Categorical(numdfT['organ'],ordered=True,categories=organ_order)
numdfT.sort_values('organ',inplace=True)
numdfT







fig=plt.figure(figsize=(4,4))
sns.boxplot(data=numdfT,x='organ',y=0,width=0.7,palette=organdict)
plt.xticks(rotation=60)
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
plt.xlabel("")
plt.ylabel("Normalied No. of activate promoters",fontsize=13)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/activate_promoter_num_organ.pdf',dpi=300,bbox_inches='tight')

groups = [group[0].values for name, group in numdfT.groupby('organ')]


f_stat, p_value = f_oneway(*groups)


p_value

groups

from itertools import combinations


numbers = range(7)
combinations_list = list(combinations(numbers, 2))
combinations_list

for i in combinations_list:
    print(i)
    print(f_oneway(groups[i[0]],groups[i[1]]))
