#!/bin/bash

ls *.fa|while read id;do(nohup fimo --max-strand -o ${id%.*}_fimo_out  mouse_JASPAR_meme.txt  $id &);done
