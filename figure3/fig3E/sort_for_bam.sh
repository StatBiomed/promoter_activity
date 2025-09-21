#!/bin/bash 
  
ls *.bam | while read id; do
    new_id="${id%.bam}.newsorted.bam"
    samtools sort -@ 20 -o "$new_id" "$id"
done
