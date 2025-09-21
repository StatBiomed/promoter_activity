#!/bin/bash

ls *.bed | while read id;do(bedtools getfasta -name -fi /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/Mus_musculus.GRCm38.dna.primary_assembly.fa -bed $id -fo ${id%.*}.fa);done
