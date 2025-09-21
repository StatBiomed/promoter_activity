rm(list = ls())


#if (!requireNamespace("BiocManager", quietly = TRUE))
#install.packages("BiocManager")
#BiocManager::install("proActiv")
#BiocManager::install("BSgenome.Mmusculus.UCSC.mm10")

library(proActiv)
library(BSgenome.Mmusculus.UCSC.mm10)


mouse.gtf.file <- '/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/ref_annotation/Mus_musculus.GRCm38.77.gtf'
mouse_annotation <- preparePromoterAnnotation(file = mouse.gtf.file, species = 'Mus_musculus')


alldir<-list.dirs('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/bam_file/Mouse',recursive = FALSE)
ovaryls<-alldir[grep('Cerebellum',alldir)]
remaiing_list <- ovaryls[!grepl("13.5", ovaryls)]
earliest_list <- ovaryls[grepl("13.5", ovaryls)]


condition_1<-rep(c(basename(earliest_list)),each=length(list.files(earliest_list)))
filename_1<-list.files(earliest_list,full.names = TRUE)


for (i in remaiing_list){
  condition_2<-rep(c(basename(i)),each=length(list.files(i)))
  final_condition<-c(condition_1,condition_2)
  
  filename_2<-list.files(i,full.names = TRUE)
  finalfile<-c(filename_1,filename_2)
  

  mouse_result <- proActiv(files = finalfile,
                           promoterAnnotation = mouse_annotation,
                           genome='mm10',
                           ncores=1,
                           condition = final_condition)
  
  result <- mouse_result[complete.cases(assays(mouse_result)$promoterCounts),]
  outputfile<-paste0('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/alternative_test_with_earliest/',refname,'_result.rds',sep='')
  saveRDS(result,outputfile)
  
  
  alternativePromoters <- getAlternativePromoters(result = result, 
                                                  referenceCondition =refname)
  
  outputfile<-paste0('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/alternative_test_with_earliest/',refname,'.rds',sep='')
  saveRDS(alternativePromoters,outputfile)
  
}

