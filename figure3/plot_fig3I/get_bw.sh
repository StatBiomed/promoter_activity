#!/bin/bash

ls *.bam|while read id;do(samtools sort -@ 20 -o ${id}_sorted $id);done
mv 2648sTS.Mouse.Liver.10.5.Female.sorted.bam_sorted 2648sTS.Mouse.Liver.10.5.Female.newsorted.bam
ls *newsorted.bam|while read id;do(nohup samtools index -@ 20 $id ${id}.bai &);done
ls *.newsorted.bam | while read id;do (nohup bamCoverage -b $id --numberOfProcessors 20 -o ${id%.*}.bw &); done
