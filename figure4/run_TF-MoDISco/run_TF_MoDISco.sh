#!/bin/bash

modisco motifs -s onehot.npz  -a all_arrtibute_score.npz -n 2000 -o modisco_results.h5
modisco report -i modisco_results.h5 -o report/ -s report/
modisco meme -i modisco_results.h5 -t PFM -o chicken_PFM.meme
