import pandas as pd

df=pd.read_excel('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/SP_MP_gene/exp_dynamic_gene.xlsx','SI Table 3',skiprows=[0,1])
df

mousedf=df[['Mouse_ID','LiverDDG']]
mousedf

dynamicdf=mousedf[mousedf['LiverDDG']==1]
dynamicdf

nondynamicdf=mousedf[mousedf['LiverDDG']==0]
nondynamicdf

classdf=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/SP_MP_gene/SP_MP_each_timepoints.csv',index_col=0)
classdf.rename(columns={'gene_id':'Mouse_ID'},inplace=True)
classdf

timepoint=set(classdf['time'])
timepoint

percentdict={}
for i in timepoint:
    print(i)
    
    selectclassdf=classdf[classdf['time']==i]
    #print(selectclassdf)
    mpdf=selectclassdf[selectclassdf['class']=='MP']
    print(mpdf)
    mp_percent=len(mpdf[mpdf['Mouse_ID'].isin(dynamicdf['Mouse_ID'])])/len(mpdf)
    spdf=selectclassdf[selectclassdf['class']=='SP']
    print(spdf)
    sp_percent=len(spdf[spdf['Mouse_ID'].isin(dynamicdf['Mouse_ID'])])/len(spdf)
    percentdict[i]=[mp_percent,sp_percent]
    

    
percentdict

percentdf=pd.DataFrame(percentdict)
percentdf

percentdf=percentdf.T
percentdf

percentdf.columns=['MP','SP']
percentdf

percentmeltdf=percentdf.melt()
percentmeltdf

percentmeltdf['variable']=pd.Categorical(percentmeltdf['variable'],categories=['SP','MP'],ordered=True)
percentmeltdf.sort_values('variable')



import seaborn as sns
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(3,4))
sns.boxplot(x='variable',y='value',data=percentmeltdf,palette=['#A9CA70','#F18C54'],width=0.3,flierprops = dict(markerfacecolor = '0.50', markersize = 2))
sns.swarmplot(data=percentmeltdf,x='variable',y='value',size=5,palette=['#A9CA70','#F18C54'],legend=False)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel("Percentage of dynamic gene",fontsize=13)
plt.xlabel("")
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/SP_MP_gene/percentage_of_dynamics_gene.pdf',dpi=300,bbox_inches='tight')







from scipy.stats import ttest_ind

ttest_ind(percentdf['MP'].values,percentdf['SP'].values)
