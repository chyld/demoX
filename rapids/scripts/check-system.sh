#!/usr/bin/bash

$echo lsb_release -a 
$echo python --version
$echo docker --version
$echo lspci | grep VGA
$echo nvidia-smi --query-gpu=driver_version --format=csv
$echo nvcc --version
