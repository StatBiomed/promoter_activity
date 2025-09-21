#!/bin/bash

samtools view -@ 20 -h -S -b -o SRR5058592_unsorted.bam SRR5058592_unsorted.sam
samtools sort -@ 10 -o SRR5058592_sorted.bam SRR5058592_unsorted.bam
sambamba view -h -t 10 -f bam -F "[XS] == null and not unmapped  and not duplicate" SRR5058592_sorted.bam > SRR5058592_sorted_unique_mapped.bam

