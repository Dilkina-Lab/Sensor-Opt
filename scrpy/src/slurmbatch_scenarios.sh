#!/bin/bash

for v in {41..80}
do
    sbatch slurmbatch.sh $v
    sleep 1
done