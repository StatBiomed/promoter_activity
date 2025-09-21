import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from pymccorrelation import pymccorrelation

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/total_promoter/Mouse_alternative_promoter_simplified.csv',index_col=0)
df

df.reset_index(inplace=True)
df

df['gene_id']=df['name'].str.split('_',expand=True)[0]
df



gexdf=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/bam_file/Mouse/Mouse.RPKM.txt',delimiter=' ')
gexdf

df=df[df['gene_id'].isin(gexdf.index.tolist())]
df

df.drop(labels=['gene_id'],axis=1,inplace=True)
df

df.set_index('name',inplace=True)
df

corrdf=df.corr()
corrdf

def get_time_df(organ):
    col_select=[i for i in corrdf.columns.tolist() if (organ in i)&('10.5' in i)]
    selectcoldf=corrdf[col_select]
    print(selectcoldf)
    selectrow=[i for i in selectcoldf.index if (organ in i)&('10.5' not in i)]
    finaldf=selectcoldf.loc[selectrow]


    
    addls=[]
    for i in finaldf.columns.tolist():
        onecol=finaldf[i]
        addls.append(onecol)
        
    allrowdf=pd.DataFrame(pd.concat(addls,axis=0))
    allrowdf.reset_index(inplace=True)
    allrowdf['time']=allrowdf['index'].str.split('.',expand=True)[3]
    allrowdf['time']=np.where(allrowdf['time'].str.contains('pb'),allrowdf['time'],allrowdf['time']+'.5')
    allrowdf['organ']=allrowdf['index'].str.split('.',expand=True)[2]
    return allrowdf
    

allorgan=[]
for i in ['Ovary','Brain','Liver','Heart','Testis']:
    oneroedf=get_time_df(i)
    allorgan.append(oneroedf)
    

allorgan

def get_time_cerebellum_df(organ):
    col_select=[i for i in corrdf.columns.tolist() if (organ in i)&('13.5' in i)]
    selectcoldf=corrdf[col_select]
    print(selectcoldf)
    selectrow=[i for i in selectcoldf.index if (organ in i)&('13.5' not in i)]
    finaldf=selectcoldf.loc[selectrow]


    
    addls=[]
    for i in finaldf.columns.tolist():
        onecol=finaldf[i]
        addls.append(onecol)
        
    allrowdf=pd.DataFrame(pd.concat(addls,axis=0))
    allrowdf.reset_index(inplace=True)
    allrowdf['time']=allrowdf['index'].str.split('.',expand=True)[3]
    allrowdf['time']=np.where(allrowdf['time'].str.contains('pb'),allrowdf['time'],allrowdf['time']+'.5')
    allrowdf['organ']=allrowdf['index'].str.split('.',expand=True)[2]
    return allrowdf
    

cerebellum=get_time_cerebellum_df('Cerebellum')
cerebellum

def get_time_kidney_df(organ):
    col_select=[i for i in corrdf.columns.tolist() if (organ in i)&('11.5' in i)]
    selectcoldf=corrdf[col_select]
    print(selectcoldf)
    selectrow=[i for i in selectcoldf.index if (organ in i)&('11.5' not in i)]
    finaldf=selectcoldf.loc[selectrow]


    
    addls=[]
    for i in finaldf.columns.tolist():
        onecol=finaldf[i]
        addls.append(onecol)
        
    allrowdf=pd.DataFrame(pd.concat(addls,axis=0))
    allrowdf.reset_index(inplace=True)
    allrowdf['time']=allrowdf['index'].str.split('.',expand=True)[3]
    allrowdf['time']=np.where(allrowdf['time'].str.contains('pb'),allrowdf['time'],allrowdf['time']+'.5')
    allrowdf['organ']=allrowdf['index'].str.split('.',expand=True)[2]
    return allrowdf
    

kidney=get_time_kidney_df('Kidney')
kidney

allorgan.append(kidney)

allorgan.append(cerebellum)

allorgan

alldf=pd.concat(allorgan,axis=0)
alldf

import seaborn as sns


custome_order=['10.5','11.5','12.5','13.5','14.5','15.5','16.5','17.5','18.5','0dpb','3dpb','2wpb','4wpb','9wpb']
custome_order

alldf['time']=pd.Categorical(alldf['time'],categories=custome_order,ordered=True)
alldf

alldf.sort_values('time',inplace=True)
alldf

fig=plt.figure(figsize=(4,5))
ax=sns.lineplot(x='time',y=0,hue='organ',data=alldf,palette=["#3399CC","#CC0000","#339900","#CC3399","#FF6600","#CC9900","#33CCFF"])
plt.xticks(rotation=90)
ax.get_legend().remove()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("")
plt.ylabel("Pearson's correlation to 10.5",fontsize=15)
plt.title('Promoter',fontsize=15)
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/promoter_line.pdf',bbox_inches='tight',dpi=300)
