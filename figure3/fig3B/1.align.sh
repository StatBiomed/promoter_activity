#!/bin/bash

bowtie2 -p 20 -q --local -x /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/mouse_bowtie_index/mouse_GRCm38 -U ./rawdata/SRR5058593.fastq.gz -S ./bowtie2_output/SRR5058593_unsorted.sam
