#!/bin/bash

ls *.newsorted.bam | while read id;do (nohup bamCoverage -b $id --numberOfProcessors 20 -o ${id%.*}.bw &); done
