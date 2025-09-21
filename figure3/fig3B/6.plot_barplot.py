import pandas as pd


4132/50851

16/106

31/186

df=pd.DataFrame({'Background':0.08125,'Up in E18.5':0.15094,'Up in 9wpb':0.16666},index=[0])
df

df=df.T
df

df.columns=['Percentage overlapping with ChIp-seq peaks']
df

df.reset_index(inplace=True)
df

df['Percentage overlapping with ChIp-seq peaks']=df['Percentage overlapping with ChIp-seq peaks']*100
df

import matplotlib.pyplot as plt


plt.bar(df['index'],df['Percentage overlapping with ChIp-seq peaks'],color=['#F1B656','#397FC7','#6F6F6F'])
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
plt.ylim(0,25)
plt.ylabel('Percentage overlapping with ChIp-seq peaks',fontsize=12)
plt.title('FOXA2')
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/figure/FOXA2_overlap_percentage.pdf',bbox_inches='tight',dpi=300)
plt.show()
