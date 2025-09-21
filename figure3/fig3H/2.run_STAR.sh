#!/bin/bash

STAR --runThreadN 10 \
        --genomeDir /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/mouse_STAR_index \
        --readFilesIn ../rawdata/SRR20993127_1.fastq.gz,../rawdata/SRR20993128_1.fastq.gz,../rawdata/SRR20993129_1.fastq.gz,../rawdata/SRR20993130_1.fastq.gz ../rawdata/SRR20993127_1.fastq.g>
        --readFilesCommand zcat \
        --outSAMtype BAM SortedByCoordinate \
        --outFileNamePrefix ./STAR_result/GSM6439101


