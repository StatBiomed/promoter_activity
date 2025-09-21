import pandas as pd

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/unibind_result/HNF4A_KO/proactiv_result/HNF4A_KO_activity.tsv',delimiter='\t')
df

df.columns=['HNF4A_KO*S1','HNF4A_KO*S2','HNF4A_KO*S3']
df

df.reset_index(inplace=True)
df

df['index']='prmtr.'+df['index'].astype('str')
df

WTdf=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/unibind_result/HNF4A_KO/proactiv_result/WT_activity.tsv',delimiter='\t')
WTdf

WTdf.columns=['WT*S1','WT*S2','WT*S3']
WTdf

WTdf.reset_index(inplace=True)
WTdf

WTdf['index']='prmtr.'+WTdf['index'].astype('str')
WTdf

alldf=df.merge(WTdf,on='index')
alldf

up9wdf=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_absolute_result/alternative_with_earliest/Up_down_geneName.csv',index_col=0)
up9wdf

upgene=up9wdf[up9wdf['classification']=='Up']
upgene

upgene['promoterId']='prmtr.'+upgene['promoterId'].astype('str')
upgene

selectdf=alldf[alldf['index'].isin(upgene['promoterId'])]
selectdf

selectdf.set_index('index',inplace=True)
selectdf

selectdf=selectdf.loc[~(selectdf==0).all(axis=1)]
selectdf

selectdf.reset_index(inplace=True)
selectdf

meltdf=selectdf.melt(id_vars=['index'])
meltdf

meltdf['condition']=meltdf['variable'].str.split('*',expand=True)[0]
meltdf

meltdf['condition']=pd.Categorical(meltdf['condition'],categories=['WT','HNF4A_KO'],ordered=True)
meltdf.sort_values('condition',inplace=True)
meltdf

import seaborn as sns
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(2,4))
sns.boxplot(data=meltdf,x='condition',y='value',palette=['#9ACE87','#EE634B'],width=0.3)
plt.ylabel('Absolute promoter activity',fontsize=14)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("")
plt.xticks(fontsize=12)
#plt.title('Up promoters',fontsize=15)
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/HNF4AKO_upregulated_promoter.pdf',bbox_inches='tight',dpi=300)
plt.show()

from scipy import stats



stats.ttest_ind(meltdf[meltdf['condition']=='WT']['value'].values,meltdf[meltdf['condition']=='HNF4A_KO']['value'].values)

