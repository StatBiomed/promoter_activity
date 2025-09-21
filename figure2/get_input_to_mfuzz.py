import pandas as pd

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/run_maSigPro/mouse/mouse_liver_signgene_fdr0_01.csv',delimiter=" ")
df

dfT=df.T
dfT

dfT.reset_index(inplace=True)
dfT

dfT['time']=dfT['index'].str.split('.',expand=True)[3]
dfT

dfT.drop(labels='index',inplace=True,axis=1)
dfT

meandf=dfT.groupby('time').mean()
meandf

meandfT=meandf.T
meandfT

customels=['10','11','12','13','14','15','16','17','18','0dpb','3dpb','2wpb','4wpb','9wpb']
customels

selectdf=meandfT[customels]
selectdf

selectdf.to_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/run_maSigPro/mouse/mouse_liver_input_to_mfuzz.csv')


