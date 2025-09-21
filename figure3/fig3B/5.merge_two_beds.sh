#!/bin/bash

bedtools intersect  -a mouse_liver_9wpb.bed -b /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/unibind_result/molecular_cell_hnf4a/macs2_output/FOXA2/FOXA2_iHep.bed  -wa > ./molecular_cell/FOXA2_overlap_9wpb.bed
