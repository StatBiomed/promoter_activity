library('clusterProfiler')

library('org.Hs.eg.db')

adata<-read.csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_result/all_samples_summary/mfuzz/used_for_GO/human/liver_C20/cluster_1.csv')
adata

conversion_result <- bitr(
  adata$geneId,
  fromType = "ENSEMBL",
  toType = "ENTREZID",
  OrgDb = org.Hs.eg.db
)

names(conversion_result)<-c('geneId','ENTREZID')
conversion_result

newadata <- merge(adata, conversion_result, by='geneId')
newadata



# GO

genes <- as.character(adata$geneId) #转换成字符格式
genes

keytypes(org.Hs.eg.db)

ego <- enrichGO(gene=genes,
                OrgDb=org.Hs.eg.db,
                keyType='ENSEMBL',
                ont='ALL',
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.05,
                qvalueCutoff  = 0.05)

ego

godf=as.data.frame(ego)
godf



finalgodf <- godf[, c("ID", "Description", "qvalue")]
finalgodf







# KEGG

kk <- enrichKEGG(gene = as.character(newadata$ENTREZID) ,
                 organism     = 'hsa',
                keyType = "kegg",
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.05,
                qvalueCutoff  = 0.05)

kkdf<-as.data.frame(kk)
kkdf

finalkeggdf<-kkdf[,c('ID','Description','qvalue')]
finalkeggdf





# Reactome

library(ReactomePA)

reactome <- enrichPathway(gene=as.character(newadata$ENTREZID), pAdjustMethod = "BH",pvalueCutoff = 0.05,qvalueCutoff  = 0.05, readable=TRUE)
reactome

reactomedf<-as.data.frame(reactome)
reactomedf



finalreactomedf<-reactomedf[,c('ID','Description','qvalue')]
finalreactomedf



finaldf=rbind(finalgodf,finalkeggdf,finalreactomedf)
finaldf

sorted_df <- finaldf[order(finaldf$qvalue), ]
sorted_df

topdf<-head(sorted_df,n=5)
topdf
