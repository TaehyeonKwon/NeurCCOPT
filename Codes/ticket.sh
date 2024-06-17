#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=48:00:00
#SBATCH --job-name=neurccopt
#SBATCH --output=%j-result.out
#SBATCH --error=%j-error.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio


module load julia

julia --project=/storage/home/tzk5446/.julia/environments/v1.8 experiments.jl 
