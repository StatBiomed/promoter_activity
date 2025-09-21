import h5py
import pandas as pd
import numpy as np
import seaborn as sns

allspecies=['Chicken','Human','Macaque','Rabbit','Rat']
allspecies

allpwmdict={}
for i in allspecies:
    inputfile='/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/'+i+'/model_explanation/modisco_results.h5'
    with h5py.File(inputfile,'r') as f:
        print(f['neg_patterns'].keys())
        # for j in f['neg_patterns'].keys():
        #     mykey=i+'_negpattern_'+j
        #     pwm=f['neg_patterns'][j]['sequence'][:].flatten()

        #     allpwmdict[mykey]=pwm

        for k in f['pos_patterns'].keys():
            mykey=i+'_'+k
            pwm=f['pos_patterns'][k]['sequence'][:].flatten()
            allpwmdict[mykey]=pwm


    

allpwmdict



for i in allpwmdict.keys():
    current_length = len(allpwmdict[i])
    if current_length < 200:
        num_zeros = 200 - current_length
        # np.concatenate 拼接两个数组
        allpwmdict[i] = np.concatenate([allpwmdict[i], np.zeros(num_zeros)])

allpwmdict



alldf=pd.DataFrame(allpwmdict).T
alldf

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

X_r = pca.fit(alldf.values).transform(alldf.values)
X_r

len(X_r)

X_rdf=pd.DataFrame(X_r)
X_rdf.columns=['PC1','PC2']
X_rdf





coldf=pd.DataFrame(alldf.index)
coldf

finaldf=pd.concat([coldf,X_rdf],axis=1)
finaldf

finaldf['species']=finaldf[0].str.split('_',expand=True)[0]
finaldf

import matplotlib.pyplot as plt
from adjustText import adjust_text

# 假设 df 已经是你的数据
plt.figure(figsize=(8,6))
plt.scatter(finaldf['PC1'], finaldf['PC2'], c='b')

# 添加标签并收集 text 对象
texts = []
for i, row in finaldf.iterrows():
    texts.append(
        plt.text(row['PC1'], row['PC2'], row[0], fontsize=9) # row[0]可替换为你想显示的列
    )

# 自动调整标签，防止重叠
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray'))

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scatter plot with separated labels')
plt.show()







sns.scatterplot(x='PC1',y='PC2',data=finaldf,hue='species')






marker_dict = {
    "Rat": "o",        # 圆形
    "Rabbit": "D",     # 菱形 ◆
    "Human": "^",      # 上三角 ▲
    "Chicken": "X",    # 叉号 ×（大写X）
    "Macaque": "P"     # 加号 +（大写P）
}

plt.figure(figsize=(3,3))
ax = sns.scatterplot(
    x='PC1',
    y='PC2',
    data=finaldf,
    markers=marker_dict,
    style='species',color='black'
)


plt.xlabel("")
plt.ylabel("")

# 去除四周边框
sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
ax.set_xticks([])
ax.set_yticks([])

handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(handles=handles, labels=labels, ncol=2, 
                   bbox_to_anchor=(2, 1), loc='upper left', borderaxespad=0.)
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/pos_pattern_pca.pdf',dpi=300,bbox_inches='tight')

plt.show()
