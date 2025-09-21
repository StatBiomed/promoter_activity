#!/bin/bash

for i in {26..30}
do
echo $i

ascp -QT -l 300m -P33001 -i /mnt/ruiyanhou/nfs_share2/conda_env/envs/PY39/etc/asperaweb_id_dsa.openssh  era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/SRR209/0${i}/SRR209931${i}/SRR209931${i}_1.fas>

ascp  -QT -l 300m -P33001 -i  /mnt/ruiyanhou/nfs_share2/conda_env/envs/PY39/etc/asperaweb_id_dsa.openssh  era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/SRR209/0${i}/SRR209931${i}/SRR209931${i}_2.f>

done
