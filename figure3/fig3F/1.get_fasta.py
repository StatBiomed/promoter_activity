import pandas as pd

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_absolute_result/run_fimo/background_fimo_out/fimo.tsv',delimiter='\t')
df

df.dropna(how='any',inplace=True)
df

df['TF']=df['motif_alt_id'].str.split('.',expand=True)[2]
df

selectdf=df[df['TF']=='Foxa2']
selectdf

twocoldf=selectdf[['sequence_name','matched_sequence']]
twocoldf

twocoldf['sequence_name']='>'+twocoldf['sequence_name']
twocoldf

twocoldf.to_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_absolute_result/run_fimo/plot_figure/background_foxa2.tsv',sep='\t',header=None,index=None)


