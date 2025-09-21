import pandas as pd
import pickle
from hilearn import CrossValidation, corr_plot
import numpy as np
import matplotlib.pyplot as plt

alldfls=[]
for i in range(0,10):
    inputfile='/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/Chicken/save_dir/TSS_true_pred_y_'+str(i)+'.pkl'
    with open(inputfile,'rb') as f:
        mydict=pickle.load(f)
    mydf=pd.DataFrame(mydict)
    alldfls.append(mydf)


    

alldf=pd.concat(alldfls,axis=0)
alldf

alldf['log_y_true'] = alldf['y_true'].apply(lambda x: np.log1p(x))

alldf['log_y_pred'] = alldf['y_pred'].apply(lambda x: np.log1p(x))


alldf





ytruenp=np.hstack(alldf['log_y_true'].values)
ytruenp

yprednp=np.hstack(alldf['log_y_pred'].values)
yprednp


plt.figure(figsize=(4,3))
corr_plot(ytruenp, yprednp, size=20,dot_color='gold')
plt.xlabel('log (observed promoter activity)',fontsize=11)
plt.ylabel('log (predicted promoter activity)',fontsize=11)
plt.title('Chicken kidney at 0dph',fontsize=14)
plt.savefig('/mnt/ruiyanhou/nfs_share2/RNA_seq_organ_species/sequence_model/kidney_all_species_birth/chicken_pearson.pdf',bbox_inches='tight',dpi=300)
plt.show()
