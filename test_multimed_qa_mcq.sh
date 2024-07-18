#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=ft-medical
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --account=kunf0097
# SBATCH --mem=80000MB
#SBATCH --output=./out/%j.out
# SBATCH --error=./out/%j.err
#SBATCH --nodelist=gpu-11-3
 
module load miniconda/3
conda activate deepeval
echo "Finally - out of queue" 
deepeval test run test_multimed_qa_mcq.py
# python test_multimed_qa_mcq.py