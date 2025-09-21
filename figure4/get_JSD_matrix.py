import pandas as pd

df=pd.read_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/conservation/PSI/psi_allspecies_alltime_removegeneID.csv',index_col=0)
df


from scipy.spatial.distance import jensenshannon


# 计算每一列之间的 Jensen-Shannon Divergence
def calculate_jsd(df):
    columns = df.columns
    jsd_matrix = pd.DataFrame(index=columns, columns=columns)

    for i in range(len(columns)):
        for j in range(i, len(columns)):
            col1 = df[columns[i]].values
            col2 = df[columns[j]].values
            
            # 计算 JSD
            jsd_value = jensenshannon(col1, col2)
            
            # 填充对称的 JSD 距离矩阵
            jsd_matrix.iloc[i, j] = jsd_value
            jsd_matrix.iloc[j, i] = jsd_value

    return jsd_matrix

# 计算并打印 JSD 距离矩阵
jsd_matrix = calculate_jsd(df)
print(jsd_matrix)

jsd_matrix

jsd_matrix = jsd_matrix.fillna(0)
jsd_matrix

jsd_matrix = jsd_matrix.astype(float)
jsd_matrix

jsd_matrix.to_csv('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/conservation/PSI/psi_JSD.csv')
