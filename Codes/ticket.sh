#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --time=24:00:00
#SBATCH --job-name=neurccopt
#SBATCH --output=result-%j.out
#SBATCH --error=error-%j.err
#SBATCH --account=azs7266_p_gpu
#SBATCH --partition=sla-prio


module load julia

julia --project=/storage/home/tzk5446/.julia/environments/v1.8 experiments.jl 
