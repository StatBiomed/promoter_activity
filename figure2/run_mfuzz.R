library(Mfuzz)

# 1. 加载您已有的数据和结果
ex <- read.csv("/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/run_maSigPro/mouse/mouse_liver_input_to_mfuzz.csv", 
               as.is=TRUE, header=TRUE, row.names=1)
ex.m <- as.matrix(ex)
eset <- new('ExpressionSet', exprs=ex.m)

eset



# 您已经运行过的Mfuzz分析
m1 <- mestimate(eset)
cl <- mfuzz(eset, c=20, m=m1)


pdf("/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/run_mfuzz/mouse/liver/C20/mouse_liver_mfuzz_C20.pdf")
mfuzz.plot2(eset, cl=cl,mfrow=c(4,4),centre=TRUE,x11=F,centre.lwd=0.2)
dev.off()

for(i in 1:20){
  potname<-names(cl$cluster[unname(cl$cluster)==i])
  write.csv(cl[[4]][potname,i],paste0("/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/run_mfuzz/mouse/liver/C20","/mfuzz_",i,".csv"))
}

# 2. 提取核心基因(α > 0.8)并计算核心表达谱
n_clusters <- 20  # 您使用了20个聚类
core_profiles <- matrix(nrow=n_clusters, ncol=ncol(ex.m))
cluster_sizes <- numeric(n_clusters)  # 记录每个聚类的核心基因数量

core_profiles



for(cluster in 1:n_clusters){
  # 获取当前聚类的成员值
  membership <- cl$membership[, cluster]
  
  # 筛选成员值>0.8的基因
  core_idx <- which(membership > 0.8)
  cluster_sizes[cluster] <- length(core_idx)
  
  if(length(core_idx) > 0){
    # 提取这些基因的表达数据并标准化
    cluster_data <- ex.m[core_idx, , drop=FALSE]
    scaled_data <- t(scale(t(cluster_data)))  # 按行标准化
    
    # 计算平均表达谱
    core_profiles[cluster, ] <- colMeans(scaled_data)
  } else {
    # 如果没有基因满足条件，设为NA
    core_profiles[cluster, ] <- rep(NA, ncol(ex.m))
  }
}

# 3. 设置行名和列名
rownames(core_profiles) <- paste0("Cluster_", 1:n_clusters, 
                                 " (n=", cluster_sizes, ")")
colnames(core_profiles) <- colnames(ex.m)

# 4. 移除没有核心基因的聚类(如果有的话)
valid_clusters <- which(!is.na(core_profiles[,1]))
core_profiles <- core_profiles[valid_clusters, ]

core_profiles

# 6. 可选: 保存核心表达谱数据
write.csv(core_profiles, 
          "/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/run_mfuzz/mouse/liver/C20/plot_heatmap/mouse_liver_C20_core_profiles.csv")


