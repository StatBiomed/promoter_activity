import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


parentfile='/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/all_samples_summary/mfuzz/used_for_heatmap/mouse'

allmousels=os.listdir(parentfile)
allmousels

for i in allmousels:
    inputfilePath=os.path.join(parentfile,i)
    df=pd.read_csv(inputfilePath)
    dfcopy=df.copy()
    df.set_index('Unnamed: 0',inplace=True)
    dfT=df.T
    dfT.reset_index(inplace=True)
    dfT['index']=dfT['index'].str.split('X',expand=True)[1]
    dfT['index']=np.where(dfT['index'].str.contains('pb'),dfT['index'],dfT['index']+'.5')
    dfT.set_index('index',drop=True,inplace=True)
    conamels=dfT.columns.tolist()
    newname=[i for i in range(1,21)]
    dfT.columns=newname
    g=sns.clustermap(dfT,row_cluster=False,cmap='coolwarm',figsize=(6.5, 5),cbar_pos=(0.6, 0.925, 0.25, 0.04),
                 cbar_kws={'orientation': 'horizontal'},  xticklabels=True)
    ax = g.ax_heatmap
    ax.set_xlabel("")
    ax.set_ylabel("")


    
    mytitle=i.split('_')[0] +' in Mouse'
    plt.title(mytitle)
    
    g.ax_heatmap.yaxis.set_ticks_position('left')  # 确保 yticks 在左侧
    mytitle=mytitle.replace(' ','_')

    outputfile=os.path.join('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/all_samples_summary/mfuzz/heatmap_fig',mytitle)
    g.savefig(outputfile,dpi=300,bbox_inches='tight')

    col_order = g.dendrogram_col.reordered_ind  # 列的重新排列索引
    clustered_columns = dfT.columns[col_order].tolist()

    selectdf=dfcopy[['Unnamed: 0']]
    selectdf['count']=selectdf['Unnamed: 0'].str.split('=',expand=True)[1].str.split(')',expand=True)[0]
    selectdf['count']=selectdf['count'].astype('float')
    selectdf['new_cluster']=selectdf['Unnamed: 0'].str.split('_',expand=True)[1].str.split(' ',expand=True)[0].astype('int')

    selectdf['new_cluster']=pd.Categorical(selectdf['new_cluster'],categories=clustered_columns,ordered=True)
    selectdf.sort_values('new_cluster',inplace=True)

    plt.figure(figsize=(5, 3.5))
    sns.barplot(data=selectdf, x="new_cluster", y="count", color='lightgray',edgecolor="black",  
        linewidth=0.5)
    plt.xlabel('',)
    plt.ylabel('Number of Promoter',fontsize=12)
    plt.xticks("")

    countoutputPath=os.path.join('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/all_samples_summary/mfuzz/bar_fig',mytitle)
    plt.savefig(countoutputPath,bbox_inches='tight',dpi=300)


    
