import pandas as pd

import os

allfiles=os.listdir('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_absolute_result/run_fimo')
allfiles

bedfiles=[i for i in allfiles if '.bed' in i]
bedfiles

for i in bedfiles:
    inputfilePath=os.path.join('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_absolute_result/run_fimo',i)
    df=pd.read_csv(inputfilePath,delimiter='\t',header=None)
    
    df[0]=df[0].str.split('chr',expand=True)[1]

    outputfile=os.path.join('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_absolute_result/run_fimo',i)
    df.to_csv(outputfile,sep='\t',header=None,index=None)
    print(df)
