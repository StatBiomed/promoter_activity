rm(list = ls())


#if (!requireNamespace("BiocManager", quietly = TRUE))
#install.packages("BiocManager")
#BiocManager::install("proActiv")

library(proActiv)
#BiocManager::install("BSgenome.Mmusculus.UCSC.mm10")
library(BSgenome.Mmusculus.UCSC.mm10)


mouse.gtf.file <- '/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/Mus_musculus.GRCm38.77.gtf'
mouse_annotation <- preparePromoterAnnotation(file = mouse.gtf.file, species = 'Mus_musculus')


alldir<-list.dirs('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/unibind_result/HNF4A_KO/bam_files',recursive = FALSE)


for (dir in alldir) {
  print(dir)
  sub_name<-basename(dir)
  print(sub_name)
  
  allfiles<-list.files(dir,full.names = TRUE)
  
  condition <- rep(c(sub_name), each=length(allfiles))
  
  mouse_result <- proActiv(files = allfiles,
                           promoterAnnotation = mouse_annotation,
                           genome='mm10',
                           ncores=1,
                           condition = condition)
  
  
  mouse_result <- mouse_result[complete.cases(assays(mouse_result)$promoterCounts),]
  
  outputfile=paste('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/unibind_result/HNF4A_KO/proactiv_result/',sub_name,'.rds',sep='')
  saveRDS(mouse_result,outputfile)
  
  
  result_df <- rowData(mouse_result)
  result_df$txId <- vapply(result_df$txId, paste, collapse = ", ", character(1L))
  outputfile=paste('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/unibind_result/HNF4A_KO/proactiv_result/',sub_name,'_rowData.tsv',sep='')
  write.table(result_df, outputfile, sep = "\t", quote = F)
  
  
  result_absolutePromoterActivity_df <- assays(mouse_result)$absolutePromoterActivity
  output_activ_file=paste('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/unibind_result/HNF4A_KO/proactiv_result/',sub_name,'_activity.tsv',sep='')
  write.table(result_absolutePromoterActivity_df,output_activ_file , sep = "\t", quote = F)
  
}



