#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --job-name=neurccopt
#SBATCH --output=result.out
#SBATCH --error=result.err
#SBATCH --account=azs7266_sc
#SBATCH --partition=sla-prio


module load julia

julia --project=/storage/home/tzk5446/.julia/environments/v1.8 experiments.jl 
