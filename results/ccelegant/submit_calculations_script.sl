#!/bin/bash

#SBATCH --job-name=elegant_tracking
#SBATCH --exclusive

##The output directory will be the current one in the console

#SBATCH --nodes=1 
#SBATCH --output=results.txt 
#SBATCH --error=errors.txt 
#SBATCH --partition=long,medium 
#SBATCH --ntasks-per-node=48 
#SBATCH --cpus-per-task=1 

set -euo pipefail

## to launch the simulation, write in the terminal: sbatch submit_calculations_script.sl
## Nodes 80-85 have 64 cores

cd "$SLURM_SUBMIT_DIR"
export RPN_DEFNS="$SLURM_SUBMIT_DIR/defns.rpn"

ml OpenMPI/4.1.4-GCC-11.3.0-cxx GSL/2.7-GCC-11.3.0 X11/20220504-GCCcore-11.3.0 FFTW/3.3.10-GCC-11.3.0  
ml libgd/2.3.3-GCCcore-11.3.0

start_time=$(date +%s)
  
elegant make_Twiss_ALBA_II.ele
elegant make_bunch_ALBA_II.ele
mpiexec --np 48 --use-hwthread-cpus Pelegant track_ALBA_II.ele 
ml SciPy-bundle/2022.05-foss-2022a matplotlib/3.5.2-foss-2022a openpyxl/3.0.10-GCCcore-11.3.0 OpenMPI/4.1.4-GCC-11.3.0-cxx
ml motif/2.3.8-GCCcore-11.3.0

# Additional plots
sddsplot -columnNames=Pass,KAverage watch.wpall -device=png -output=E_vs_pass.png -graphic=line
sddsplot -columnNames=Pass,el watch.wp0 -device=png -output=el_vs_pass.png -graphic=line
sddsplot -columnNames=Pass,ecy watch.wp0 -device=png -output=ecy_vs_pass.png -graphic=line
sddsplot -columnNames=Pass,ecx watch.wp0 -device=png -output=ecx_vs_pass.png -graphic=line
 
# Additional plots if RFMODE was used
if [ -f "track_ALBA_II.rfmc" ]; then
	sddsplot -columnNames=Pass,VCavity track_ALBA_II.rfmc -device=png -output=V_c_MC_vs_pass.png -graphic=line
	sddsplot -columnNames=Pass,V track_ALBA_II.rfmc -device=png -output=V_beam_MC_vs_pass.png -graphic=line
	sddsplot -columnNames=Pass,VCavity track_ALBA_II.rfhc -device=png -output=V_c_HC_vs_pass.png -graphic=line
	sddsplot -columnNames=Pass,V track_ALBA_II.rfhc -device=png -output=V_beam_HC_vs_pass.png -graphic=line       
fi 
 
python3 plot_results_script_v_1_3.py
 
end_time=$(date +%s)
duration=$((end_time - start_time))

# Calculate duration in hours
hours=$(echo "scale=2; $duration / 3600" | bc)

echo "Total execution time: $hours hours"