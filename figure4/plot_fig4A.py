import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/conservation/PSI/psi_JSD.csv',index_col=0)
df

df.index = df.index.str.replace('Forebrain', 'Brain').str.replace('WholeBrain', 'Brain').str.replace('Hindbrain', 'Cerebellum')
df.columns = df.columns.str.replace('Forebrain', 'Brain').str.replace('WholeBrain', 'Brain').str.replace('Hindbrain', 'Cerebellum')


# 提取信息并创建颜色映射
species = []
organ = []
development_time = []

for label in df.index:
    parts = label.split('_')
    species.append(parts[0].lower())
    organ.append(parts[1].lower())
    development_time.append(parts[2])

set(species)

set(organ)

development_time







# 定义颜色映射
species_color_map = {
    'human': 'red',
    'macaque': 'orange',
    'mouse': 'gray',
    'rat': 'black',
    'rabbit': 'brown',
    'chicken': 'violet'
}

organ_color_map = {
    'brain': '#3399CC',
    'cerebellum': '#33CCFF',
    'heart': '#CC0000',
    'kidney': '#CC9900',
    'liver': '#339900',
    'ovary': '#CC3399',
    'testis': '#FF6600'
}

# 创建 row_colors DataFrame
row_colors = pd.DataFrame({
    'Species': [species_color_map[s] for s in species],
    'Organ': [organ_color_map[o] for o in organ],
}, index=df.index)
row_colors

import numpy as np



# 绘制 clustermap
g = sns.clustermap(df, row_colors=row_colors, col_cluster=True, row_cluster=True,xticklabels=False,yticklabels=False,dendrogram_ratio=(0.1, 0),cbar_pos=(0.996, 0.08, 0.015, 0.9))
#plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)


# 创建物种图例
# species_legend_patches = [mpatches.Patch(color=color, label=label) for label, color in species_color_map.items()]
# print(species_legend_patches)
# species_legend = plt.legend(handles=species_legend_patches,ncol=2 , bbox_to_anchor=(18.5, 1.105))

# 创建器官图例
#organ_legend_patches = [mpatches.Patch(color=color, label=label) for label, color in organ_color_map.items()]
#organ_legend = plt.legend(handles=organ_legend_patches,ncol=1, bbox_to_anchor=(4.7, 0.96))

# 添加图例到当前轴
#plt.gca().add_artist(species_legend)
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/conservation/PSI/PSI_JSD_species_organ.pdf',bbox_inches='tight',dpi=300)

plt.show()
