#!/bin/bash

ls *newsorted.bam|while read id;do(nohup samtools index -@ 20 $id ${id}.bai &);done
