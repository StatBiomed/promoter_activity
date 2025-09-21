import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from pymccorrelation import pymccorrelation
import os

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/alternative_test_with_earliest/Up_down_geneName.csv',index_col=0)
df


df['organ']=df['origin'].str.split('_',expand=True)[1]
df

liverdf=df[df['organ']=='Liver']
liverdf

liverdf.sort_values('padjAbs',inplace=True)
liverdf

liverdf[liverdf['geneId']=='ENSMUSG00000018819']



liver_selectdf=liverdf.head(n=50)
liver_selectdf













updf=liver_selectdf[liver_selectdf['classification']=='Up']
updf

downdf=liver_selectdf[liver_selectdf['classification']=='Down']
downdf



upgene=set(updf['geneId'])
upgene

downgene=set(downdf['geneId'])
downgene

upgene & downgene





liverdf[liverdf['geneId']=='ENSMUSG000000018819']







signpromoterls=list(set(liver_selectdf['promoterId']))
signpromoterls

len(signpromoterls)

signpromoterls=[str(i) for i in signpromoterls]
signpromoterls





# Get total gene

activitydf=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/total_promoter/Mouse_total_promoter_simplified.csv',index_col=0)
activitydf


heartcol=[i for i in activitydf.columns.tolist() if 'Liver' in i]
heartcol

selectdf=activitydf[heartcol]
selectdf

selectdf.reset_index(inplace=True)
selectdf

selectdf['promoterId']=selectdf['name'].str.split('_',expand=True)[4].str.split('ProID',expand=True)[1]
selectdf

signdf=selectdf[selectdf['promoterId'].isin(signpromoterls)]
signdf

signdf['geneID']=signdf['name'].str.split('_',expand=True)[0]
signdf

gennameconverdf=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/mouse_gene_name_transfer.csv',index_col=0)
gennameconverdf.columns=['geneID','gene_name']
gennameconverdf

signdf=signdf.merge(gennameconverdf,on='geneID')
signdf

signdf['promoterId']='prmtr.'+signdf['promoterId']+'_'+signdf['gene_name']
signdf



signdf.drop(labels=['name','geneID','gene_name'],inplace=True,axis=1)
signdf

signdf.set_index('promoterId',inplace=True)
signdf

import seaborn as sns

colls=signdf.columns.tolist()
colls

mydf=pd.DataFrame(colls)
mydf

mydf['time']=mydf[0].str.split('.',expand=True)[3]
mydf

mydf['time']=np.where(mydf['time'].str.contains('pb'),mydf['time'],mydf['time']+'.5')
mydf

custome_order=['10.5','11.5','12.5','13.5','14.5','15.5','16.5','17.5','18.5','0dpb','3dpb','2wpb','4wpb','9wpb']

mydf['time']=pd.Categorical(mydf['time'],ordered=True,categories=custome_order)

mydf.sort_values(by='time',inplace=True)
mydf

finalsigndf=signdf[mydf[0].tolist()]
finalsigndf

import colorcet as cc

lut={'10.5':'#e64b35cc','11.5':'#4dbbd5cc','12.5':'#00a087cc','13.5':'#9fdaf7',
    '14.5':'#f39b7fcc','15.5':'#8491b4cc','16.5':'#91d1c2cc','17.5':'#57919d','18.5':'#7e6148cc',
    '0dpb':'#d9dee7','3dpb':'#ecb426','2wpb':'#416594','4wpb':'#f9b3ad','9wpb':'#c9a1ca'}





# lut = dict(zip(set(mydf['time']), sns.color_palette(cc.glasbey,len(set(mydf['time'])))))
# lut

mydf['time']=mydf['time'].astype('object')

mydf['time']

col_colors=mydf['time'].map(lut)
col_colors








ax=sns.clustermap(finalsigndf,cmap='GnBu',xticklabels=False,yticklabels=True,row_cluster=True,col_cluster=False,col_colors=[col_colors],
              figsize=(12,18),cbar_kws=dict(orientation='horizontal'))
ax.ax_heatmap.set_ylabel("")
ax.tick_params(axis='both', which='both', length=0)
ax.ax_heatmap.tick_params(axis='y', labelsize=18)  # y 轴刻度字体大小

x0, _y0, _w, _h = ax.cbar_pos
ax.ax_cbar.set_position([0.2, -0.04, ax.ax_row_dendrogram.get_position().width, 0.02])

ax.ax_cbar.tick_params(axis='x', length=5)  #设置cbar引出的line的长度
for spine in ax.ax_cbar.spines:              #为cbar设置边框
    ax.ax_cbar.spines[spine].set_color('black')
    ax.ax_cbar.spines[spine].set_linewidth(1)

plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/liver_abs_activity_sign_top50.pdf',dpi=300,bbox_inches='tight')
plt.show()



promoterID=[i.split('.')[1].split('_')[0] for i in list(finalsigndf.index)]
promoterID



liver_selectdf



for i in promoterID:
    print(i)
    selectdf=liver_selectdf[liver_selectdf['promoterId']==int(i)]
    print(selectdf)
    print("***************************************************")
    
