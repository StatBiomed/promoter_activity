from kipoiseq.transforms.functional import one_hot
from pyfaidx import Fasta
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from scipy import stats
import torch.optim as optim
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler
import h5py
import random
import os
import pickle


def get_onehot_withchr(fasta,bed):
    ref_fastq=fasta
    bed=bed
    beddf=pd.read_csv(bed,index_col=0)
    genes=Fasta(ref_fastq)
    beddf['extract_start']=beddf['start']-10000
    beddf['extract_end']=beddf['start']+10000
    beddf=beddf[beddf['extract_start']>0]

    #extract sequence and do one-hot encoding 
    onehotdict={}
    for chrom, start, end, strand, genename in tqdm(beddf[['seqnames','extract_start','extract_end','strand','promoterId']].values,desc='Loading Loci'):
        if strand=='+':
            seq_=genes.get_seq(chrom,start+1,end).seq
        else:
            seq_=genes.get_seq(chrom,start+1,end,rc=True).seq
    
        onehotnp=one_hot(seq_,['A','C','T','G'])
        onehotdict[genename]=onehotnp

    outputPath=os.path.join(os.path.dirname(bed),'protein_coding_promoter_onehot.pkl')
    with open(outputPath,'wb') as f:
        pickle.dump(onehotdict,f)
    

        

def get_onehot_withoutchr(fasta,bed):
    ref_fastq=fasta
    bed=bed
    beddf=pd.read_csv(bed,index_col=0)
    beddf['seqnames']=beddf['seqnames'].str.split('chr',expand=True)[1]
    print(beddf)
    genes=Fasta(ref_fastq)
    beddf['extract_start']=beddf['start']-10000
    beddf['extract_end']=beddf['start']+10000
    beddf=beddf[beddf['extract_start']>0]

    #extract sequence and do one-hot encoding 
    onehotdict={}
    for chrom, start, end, strand, genename in tqdm(beddf[['seqnames','extract_start','extract_end','strand','promoterId']].values,desc='Loading Loci'):
        if strand=='+':
            seq_=genes.get_seq(chrom,start+1,end).seq
        else:
            seq_=genes.get_seq(chrom,start+1,end,rc=True).seq
    
        onehotnp=one_hot(seq_,['A','C','T','G'])
        onehotdict[genename]=onehotnp

    outputPath=os.path.join(os.path.dirname(bed),'protein_coding_promoter_onehot.pkl')
    with open(outputPath,'wb') as f:
        pickle.dump(onehotdict,f)
    

        

fastals=['/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/fasta/Homo_sapiens/GRCh37.primary_assembly.genome.fa',
        '/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/fasta/Mus_musculus/GRCm38.primary_assembly.genome.fa',
        ]

onespeciesls=['Human','Mouse']
onebedfilels=['/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/'+i+'/protein_coding_promoter.csv' for i in onespeciesls ]
onebedfilels



otherfastals=['/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/fasta/Gallus_galgal4/Gallus_gallus.Galgal4.dna.toplevel.fa',
              '/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/fasta/Macaca_mulatta/Macaca_mulatta.MMUL_1.dna.toplevel.fa',
              '/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/fasta/Oryctolagus_cuniculus/Oryctolagus_cuniculus.OryCun2.0.dna.toplevel.fa',
            '/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/fasta/Rattus_norvegicus/Rattus_norvegicus.Rnor_5.0.dna.toplevel.fa'
]



species=['Chicken','Macaque','Rabbit','Rat']

bedfilels=['/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/'+i+'/protein_coding_promoter.csv' for i in species ]
bedfilels

for i,j in zip(fastals,onebedfilels):
    get_onehot_withchr(i,j)
    

for z,k in zip(otherfastals,bedfilels):
    print(z)
    print(k)
    get_onehot_withoutchr(z,k)
