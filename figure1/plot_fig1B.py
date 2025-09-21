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
    allls.append(valuedf)

alldf=pd.concat(allls,axis=0)
alldf

alldf.to_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/sample_major_minor_count.csv')

import pandas as pd

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/sample_major_minor_count.csv')
df

custome_order=['Major','Minor','Inactive']

df['Unnamed: 0']=pd.Categorical(df['Unnamed: 0'],categories=custome_order,ordered=True)
df

df.sort_values('Unnamed: 0',inplace=True)
df

df['count']=df['count']/1000
df

import seaborn as sns
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(4,5))
sns.violinplot(data=df,x='Unnamed: 0',y='count',palette=['#FF5E00','#F3D707','#5CB3BA'])
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Promoter type",fontsize=13)
plt.ylabel(" ",fontsize=13)
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/mouse_promoter_type.pdf',bbox_inches='tight',dpi=300)



majordf=df[df['Unnamed: 0']=='Major']
majordf

minordf=df[df['Unnamed: 0']=='Minor']
minordf

inactivatedf=df[df['Unnamed: 0']=='Inactive']
inactivatedf



from scipy.stats import ttest_ind

ttest_ind(majordf['count'].values,minordf['count'].values)

ttest_ind(minordf['count'].values,inactivatedf['count'].values)

ttest_ind(majordf['count'].values,inactivatedf['count'].values)
