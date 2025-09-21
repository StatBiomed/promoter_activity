import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/total_promoter/Mouse_total_promoter_simplified.csv',index_col=0)
df

corrdf=df.corr()
corrdf

X=corrdf.values
X

from sklearn.manifold import MDS

embedding = MDS(n_components=2, normalized_stress='auto',random_state=666)

X_transformed = embedding.fit_transform(X)

X_transformed

mdsdf=pd.DataFrame(X_transformed,columns=['MDS_dim1','MDS_dim2'])
mdsdf

sampledf=pd.DataFrame(corrdf.index)
sampledf

alldf=pd.concat([sampledf,mdsdf],axis=1)
alldf

alldf['time']=alldf[0].str.split('.',expand=True)[3]
alldf['time']=np.where(alldf['time'].str.contains('pb'),alldf['time'],alldf['time']+'.5')
alldf['organ']=alldf[0].str.split('.',expand=True)[2]
alldf

set(alldf['time'])

transferdf=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/time_transfer/Mouse.csv',index_col=0)
transferdf.columns=['time','new_time_days']
transferdf

finaldf=alldf.merge(transferdf,on='time')
finaldf

finaldf


palettedict={'Ovary':"#CC3399",'Liver':"#339900",'Heart':"#CC0000",'Brain':"#3399CC",'Testis':"#FF6600",
            'Kidney':"#CC9900",'Cerebellum':"#33CCFF"}

fig,ax=plt.subplots()

g=sns.scatterplot(x='MDS_dim1',y='MDS_dim2',data=finaldf,hue='organ',size='new_time_days',
               palette=palettedict,s=60)
#plt.legend(loc=(0.03,0),title="")

handles, labels = ax.get_legend_handles_labels()  # 获取当前图例的句柄和标签
#ax.legend(handles=handles[1:], labels=labels[1:])
organ_legend = handles[1:len(finaldf['organ'].unique())+1]  # 提取 organ 图例的句柄
organ_labels = labels[1:len(finaldf['organ'].unique())+1]  # 提取 organ 图例的标签
plt.legend(organ_legend, organ_labels,loc=(0.04, 0.6),title=None,ncol=2)  # 去掉标题并设置位置
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Dim 1',fontsize=14)
plt.ylabel('Dim 2',fontsize=14)
plt.title('Mouse',fontsize=15)
plt.gca().spines['top'].set_visible(False) 
plt.gca().spines['right'].set_visible(False)
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/1.PC_total_promoter.pdf',bbox_inches='tight',dpi=300)
plt.show()
