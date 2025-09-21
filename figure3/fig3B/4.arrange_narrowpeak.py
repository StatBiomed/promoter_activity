import pandas as pd

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/unibind_result/molecular_cell_hnf4a/macs2_output/FOXA2/FOXA2_peaks.narrowPeak',delimiter='\t',header=None)
df

df.columns=['chr','start','end','peak_name','-10*log10(qvalue)','strand','fold_change','-log_10_pvalue','-log_10_qvalue','summit_position_relative_to_start']
df

df[df['-log_10_qvalue']>1.3]

df['score']=0
df

beddf=df[['chr','start','end','peak_name','score','strand']]
beddf

beddf['chr']='chr'+beddf['chr']
beddf

beddf.to_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/unibind_result/molecular_cell_hnf4a/macs2_output/FOXA2/FOXA2_iHep.bed',sep='\t',header=None,index=None)


