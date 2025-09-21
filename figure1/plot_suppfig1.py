import pandas as pd
import os

allfile=os.listdir('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/Mouse')
allfile

rowfile=[i for i in allfile if 'rowData' in i]
rowfile

allls=[]
for i in rowfile:
    inputfile=os.path.join('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/Mouse/',i)
    df=pd.read_csv(inputfile,index_col=0,delimiter='\t')
    col=i.split('_rowData')[0]+'.class'
    valuedf=pd.DataFrame(df[col].value_counts())
    print(valuedf)
    valuedf['sample']=i
    allls.append(valuedf)

alldf=pd.concat(allls,axis=0)
alldf

alldf['time']=alldf['sample'].str.split('_rowData.tsv',expand=True)[0].str.split('_',expand=True)[2]
alldf

alldf['organ']=alldf['sample'].str.split('_',expand=True)[1]
alldf

liverdf=alldf[alldf['organ']=='Liver']
liverdf

liverdf.reset_index(inplace=True)
liverdf



custome_order=['Major','Minor','Inactive']

liverdf['index']=pd.Categorical(liverdf['index'],categories=custome_order,ordered=True)
liverdf

liverdf.sort_values('index',inplace=True)
liverdf

liverdf['count']=liverdf['count']/1000
liverdf

import seaborn as sns
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(4,5))
sns.violinplot(data=liverdf,x='index',y='count',palette=['#FF5E00','#F3D707','#5CB3BA'])
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("")
plt.ylabel("")
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/mouse_promoter_type_Liver.pdf',bbox_inches='tight',dpi=300)



from scipy.stats import ttest_ind
from scipy.stats import f_oneway

ttest_ind(liverdf[liverdf['index']=='Major']['count'].values,liverdf[liverdf['index']=='Minor']['count'].values)

ttest_ind(liverdf[liverdf['index']=='Minor']['count'].values,liverdf[liverdf['index']=='Inactive']['count'].values)

ttest_ind(liverdf[liverdf['index']=='Major']['count'].values,liverdf[liverdf['index']=='Inactive']['count'].values)



f_oneway(liverdf[liverdf['index']=='Major']['count'].values,liverdf[liverdf['index']=='Minor']['count'].values,liverdf[liverdf['index']=='Inactive']['count'].values)










pb9wdf=alldf[alldf['time']=='9wpb']
pb9wdf

pb9wdf.reset_index(inplace=True)
pb9wdf

pb9wdf['count']=pb9wdf['count']/1000
pb9wdf

pb9wdf['index']=pd.Categorical(pb9wdf['index'],categories=custome_order,ordered=True)
pb9wdf

pb9wdf.sort_values('index',inplace=True)
pb9wdf



fig=plt.figure(figsize=(4,5))
sns.violinplot(data=pb9wdf,x='index',y='count',palette=['#FF5E00','#F3D707','#5CB3BA'])
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("")
plt.ylabel("")
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/mouse_promoter_type_9wpb.pdf',bbox_inches='tight',dpi=300)



ttest_ind(pb9wdf[pb9wdf['index']=='Major']['count'].values,pb9wdf[pb9wdf['index']=='Minor']['count'].values)

ttest_ind(pb9wdf[pb9wdf['index']=='Minor']['count'].values,pb9wdf[pb9wdf['index']=='Inactive']['count'].values)

ttest_ind(pb9wdf[pb9wdf['index']=='Major']['count'].values,pb9wdf[pb9wdf['index']=='Inactive']['count'].values)

from scipy.stats import f_oneway

f_oneway(pb9wdf[pb9wdf['index']=='Major']['count'].values,pb9wdf[pb9wdf['index']=='Minor']['count'].values,pb9wdf[pb9wdf['index']=='Inactive']['count'].values)
