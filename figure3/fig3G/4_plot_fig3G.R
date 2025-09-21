rm(list=ls())

# 加载ggplot2包
library(ggplot2)

# 创建数据框
df <- data.frame(
  condition = c("background", "18.5", "9wpb", "background", "18.5", "9wpb"),
  percentage = c(50.86, 61.32, 57.53, 55.42, 67.92, 65.59),
  TF = c("FOXA2", "FOXA2", "FOXA2", "HNF4A", "HNF4A", "HNF4A")
)

# 将 condition 列因子化并设置级别顺序
df$condition <- factor(df$condition, levels = c("background", "18.5", "9wpb"))

# 设置颜色
col_set <- c("background" = "#ffb25f", "18.5" = "#bab4d0", "9wpb" = "#b57aaa")

# 绘制棒棒糖图
p <- ggplot(df, aes(x = condition, y = percentage)) +
  # 棒棒图的连线
  geom_segment(aes(x = condition, xend = condition, y = 0, yend = percentage),
               linetype = "solid", # 实线
               size = 1, # 连线的粗细
               color = "gray40" # 连线的颜色
  ) +
  # y轴0刻度的水平线
  geom_hline(
    yintercept = 0,  # 水平线位置
    linetype = "dashed",  # 虚线
    size = 1, # 连线的粗细
    colour = "gray40" # 连线的颜色
  ) +
  # 绘制点
  geom_point(aes(color = condition),
             size = 8) +   
  # 添加相关性R值标签
  geom_text(aes(label = percentage),
            color = "black",
            size = 4,
            hjust=0.5,
            vjust = 0.5) +
  # y轴刻度设置
  scale_y_continuous( # 设置y轴
    limits = c(0, 70),
    breaks = seq(0, 70, by = 10),
    labels = seq(0, 70, by = 10)) +
  # 设置颜色
  scale_color_manual(values = col_set) +
  # 坐标title和图形title设置
  labs(
    y = "Percentage",
    #title = "Binding Percentage by Condition"
  ) +
  theme_classic() +
  theme(
    plot.title = element_text(size = 13, hjust = 0.5),
    axis.text.x = element_text(size = 13, angle = 45, hjust = 1, face = 'bold'),
    axis.text.y = element_text(size = 13, face = 'bold'),
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 13, face = 'bold'),
    plot.margin = unit(c(1, 1, 1, 1), "cm"),
    legend.position = 'none',
    strip.text = element_text(size = 13, face = "bold")
  ) +
  # 分面
  facet_wrap(~TF, scales = "free_x")

# 显示图形
print(p)

# 保存图形
ggsave(plot = p,
       device = 'png',
       dpi = 300,
       width = 5,
       height = 5,
       filename = '/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/proactiv_absolute_result/run_fimo/plot_figure/plot.png')


