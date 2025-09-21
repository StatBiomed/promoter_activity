#!/bin/bash
  
make_tracks_file --trackFiles /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/example_gene_gtf/Dcakd.gtf \
        /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/example/2583sTS.Mouse.Liver.10.5.Female.newsorted.bw \
        /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/example/2648sTS.Mouse.Liver.10.5.Female.newsorted.bw \
        /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/example/2496sTS.Mouse.Liver.10.5.Male.newsorted.bw \
        /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/example/3461sTS.Mouse.Liver.10.5.Female.newsorted.bw \
        /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/example/1946sTS.Mouse.Liver.9wpb.Male.sorted.newsorted.bw \
        /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/example/1952sTS.Mouse.Liver.9wpb.Female.sorted.newsorted.bw \
        /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/example/1958sTSm.Mouse.Liver.9wpb.Male.sorted.newsorted.bw \
        /mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/example/1962sTS.Mouse.Liver.9wpb.Female.sorted.newsorted.bw \
        -o Dcaked_track.ini
