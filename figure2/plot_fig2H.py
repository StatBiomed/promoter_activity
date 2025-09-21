import pandas as pd
import numpy as np

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/example/Gene_exp_10.5_0dpb.csv',index_col=0)
df

geneexp=pd.DataFrame(df.loc[5273])
geneexp

geneexp.reset_index(inplace=True)
geneexp

geneexp['time']=geneexp['index'].str.split('.',expand=True)[3]
geneexp

geneexp['time']=np.where(geneexp['time'].str.contains('pb'),geneexp['time'],geneexp['time']+'.5')
geneexp

from scipy.stats import ttest_ind 

ttest_ind(geneexp[geneexp['time']=='10.5'][5273].values,geneexp[geneexp['time']=='0dpb'][5273].values)

import seaborn as sns
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(2,4))
sns.boxplot(data=geneexp,x='time',y=5273.0,palette=['#00A664','#00A664'],width=0.3)
plt.ylabel('Gene expression',fontsize=14)
ax=plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("")
plt.xticks(fontsize=12)
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/Lsp1_gene_exp.pdf',bbox_inches='tight',dpi=300)
#plt.title('prmtr.5272',fontsize=15)
