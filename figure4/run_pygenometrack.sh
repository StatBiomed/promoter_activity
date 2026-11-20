#!/bin/bash
  
make_tracks_file --trackFiles /mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/Fpgs.gtf \
	/mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/SRX20201036_HNF4A_Mouse_Liver.bw \
	/mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/SRX019010_E2F4_Mouse_Liver.bw \
        /mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.10.5.mean_n4.bw \
        /mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.11.5.mean_n5.bw \
        /mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.12.5.mean_n4.bw \
        /mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.13.5.mean_n5.bw \
        /mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.14.5.mean_n4.bw \
        /mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.15.5.mean_n4.bw \
        /mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.16.5.mean_n4.bw \
        /mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.17.5.mean_n4.bw \
	/mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.18.5.mean_n4.bw \
	/mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.0dpb.mean_n4.bw \
	/mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.3dpb.mean_n4.bw \
	/mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.2wpb.mean_n4.bw \
	/mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.4wpb.mean_n4.bw \
	/mnt/ruiyanhou/nfs_share2/proactiv_redo/13.promoter_usage/6.plot_example/merge_same_stage/Mouse.Liver.9wpb.mean_n4.bw \
        -o Fpgs_track.ini
