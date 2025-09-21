import pandas as pd

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/run_mfuzz/mouse/liver/C20/plot_heatmap/mouse_liver_C20_core_profiles.csv')

df

selectdf=df[['Unnamed: 0']]
selectdf

selectdf['count']=selectdf['Unnamed: 0'].str.split('=',expand=True)[1].str.split(')',expand=True)[0]
selectdf

selectdf['count']=selectdf['count'].astype('float')
selectdf

selectdf['new_cluster']=selectdf['Unnamed: 0'].str.split('_',expand=True)[1].str.split(' ',expand=True)[0].astype('int')
selectdf

newcluster=[20, 11, 1, 2, 12, 10, 17, 8, 13, 4, 5, 7, 16, 14, 9, 18, 19, 15, 3, 6]
newcluster



selectdf['new_cluster']=pd.Categorical(selectdf['new_cluster'],categories=newcluster,ordered=True)
selectdf.sort_values('new_cluster',inplace=True)
selectdf

color_list = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",  # 经典配色（蓝、橙、绿、红、紫）
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",  # 柔和扩展（棕、粉、灰、橄榄、青）
    "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5",  # 浅色变体（浅蓝、浅橙、浅绿、浅红、浅紫）
    "#C49C94", "#F7B6D2", "#C7C7C7", "#DBDB8D", "#9EDAE5",  # 低饱和度（浅棕、浅粉、浅灰、浅黄、浅青）
    "#393B79", "#637939", "#8C6D31", "#843C39", "#7B4173",  # 深色变体（深蓝、深绿、深棕、深红、深紫）
]


nature_colors = [
    # 基础色系（高对比度，核心数据）
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",  # 红/蓝/绿/深蓝/珊瑚
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",  # 灰蓝/薄荷/正红/棕/米灰
    
    # 扩展色系（辅助数据）
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#56B4E9",  # 黄/深蓝/橙/粉/天蓝
    "#009E73", "#999999", "#E69F00", "#332288", "#882255",  # 深绿/灰/金/紫/酒红
    
    # 特殊强调色
    "#AA4499", "#117733", "#44AA99", "#88CCEE", "#DDCC77"   # 紫红/森林绿/青绿/浅蓝/沙黄
]

plt.figure(figsize=(5, 3.5))
sns.barplot(data=selectdf, x="new_cluster", y="count", palette=nature_colors,edgecolor="black",  
    linewidth=0.5)
plt.xlabel('',)
plt.ylabel('Number of Promoter',fontsize=12)
plt.xticks("")
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/run_mfuzz/mouse/liver/C20/plot_heatmap/c20_mouse_liver_count_barplot.pdf',bbox_inches='tight',dpi=300)


