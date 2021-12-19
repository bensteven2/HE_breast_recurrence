#!/bin/bash

dir_maf="/data2/ben/data/zhongzhong/breast_torch_1000/1/*"

maf_files=$(ls -d $dir_maf)

for maf_file in ${maf_files}
do
	echo "maf_file: ${maf_file}"
	cp -r  ${maf_file} ${maf_file}_1
done
