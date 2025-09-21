import pandas as pd

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/all_samples_summary/mfuzz/GO/human_liver.csv')
df

duplicates = df[df.duplicated(['Description', 'cluster'], keep=False)]
duplicates

duplicates.drop_duplicates('ID')

keepdf=df[df['ID']!='GO:0006364']
keepdf





pivot_df = keepdf.pivot(index='Description', columns='cluster', values='qvalue')
pivot_df

pivot_df.fillna(0.05,inplace=True)
pivot_df

newlist=['cluster_8', 'cluster_12', 'cluster_6', 'cluster_13', 'cluster_14', 'cluster_18', 
 'cluster_15', 'cluster_19','cluster_2','cluster_16', 'cluster_9', 'cluster_4', 'cluster_10', 'cluster_20',
 'cluster_1', 'cluster_11', 'cluster_7', 'cluster_17', 'cluster_3', 'cluster_5']
newlist

len(newlist)

descriptions = [
    "Processing of Capped Intron-Containing Pre-mRNA",
    "chromosome segregation",
    "mRNA Splicing - Major Pathway",
    "cadherin binding",
    "cell leading edge",
    "chromosomal region",
    "mRNA Splicing",
    "mitotic sister chromatid segregation",
    "ncRNA processing",
    "nuclear chromosome segregation",
    "rRNA metabolic process",
    "rRNA processing",
    "sister chromatid segregation",
    "carboxylic acid catabolic process",
    "organic acid catabolic process",
    "small molecule catabolic process",
    "fatty acid metabolic process",
    "immunoglobulin complex",
    "xenobiotic metabolic process"
]

pivot_df=pivot_df.loc[descriptions,newlist]
pivot_df

pivot_df.columns = pivot_df.columns.str.replace('cluster_', '')
pivot_df



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

heatmap_data = -np.log10(pivot_df.replace(0, 1e-30))  # 避免log(0)

# 2. 创建图形
plt.figure(figsize=(8, 5))  # 调整图形大小

# 3. 绘制热图
ax = sns.heatmap(
    heatmap_data,
    cmap="YlGnBu",  # 反向的黄色-绿色-蓝色，小值深色
    linewidths=0.5,    # 设置细边框宽度
    linecolor='white', # 边框颜色
    square=True,       # 使每个单元格为正方形
    annot=False,
    #yticklabels=False,# 不显示数值（若需要可改为True）
    cbar_kws={
        'label': '-log10(q-value)',  # 颜色条标签
        'shrink': 0.8  # 颜色条大小
    }
)

# 4. 美化图形
ax.set_xlabel('', fontsize=12)
ax.set_ylabel('', fontsize=12)

# 隐藏y轴刻度线（竖线）
ax.tick_params(
    axis='y',          # 操作y轴
    which='both',      # 同时影响主刻度和小刻度
    left=False,        # 隐藏左边的刻度线
    length=0           # 设置刻度线长度为0
)

ax.tick_params(
    axis='x',          # 操作y轴
    which='both',      # 同时影响主刻度和小刻度
    left=False,        # 隐藏左边的刻度线
    length=0           # 设置刻度线长度为0
)

# 旋转x轴标签
#plt.xticks( ha='right', rotation_mode='anchor')

# 调整布局防止标签被截断
plt.tight_layout()

# 5. 保存高清图片（可选）
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/all_samples_summary/mfuzz/GO/human_liver.pdf', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()
