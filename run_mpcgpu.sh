#!/bin/bash
#SBATCH --job-name=mpcgpu_job
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=mpcgpu_%j.out
#SBATCH --error=mpcgpu_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felix.muehlenberend@mailbox.org

# Load necessary modules (if required)
module load devel/cuda

LOG_GPU=0
SCRIPT=""

while [[ $# -gt 0 ]]; do
	case $1 in
		--sample)
			LOG_GPU=1
			shift # Remove --sample from processing
			;;
		--help|-h)
			echo "Usage: $0 [--sample] <script>"
			echo "Options:"
			echo "  --sample    Enable GPU memory sampling"
			echo "  --help, -h  Show this help message"
			exit 0
			;;
		--) # Pass arguments to python script
			shift
			break
			;;
		*)
			if [[ -n "$SCRIPT" ]]; then
				echo "Error: Multiple scripts provided. Only one script can be run at a time."
				exit 1
			fi
			SCRIPT="$1"
			shift # Remove the script name from processing
			break # Stop processing further arguments
			;;
	esac
done

# Ensure a script is provided
if [[ -z "$SCRIPT" ]]; then
	echo "Error: No script provided to run."
	echo "Usage: $0 [--sample] <script>"
	exit 1
fi

SAMPLE_MS=500

RAW_LOG="gpu_memory_${SLURM_JOB_ID}.csv"
SUMMARY_LOG="gpu_memory_summary_${SLURM_JOB_ID}.txt"
PMON_LOG="${SLURM_JOB_ID}.pmon.csv"

start_gpu_sampler () {
    [[ $LOG_GPU -eq 0 ]] && return

    (
        echo "timestamp,mem_used_MiB"
        stdbuf -oL nvidia-smi --query-gpu=timestamp,memory.used \
                              --format=csv,noheader,nounits \
                              --loop-ms="$SAMPLE_MS"
    ) >> "$RAW_LOG" &
    GPU_SAMPLER_PID=$!
}

start_proc_monitor () {
    [[ $LOG_GPU -eq 0 ]] && return
    stdbuf -oL nvidia-smi pmon -s u -d 1 -o T >> "$PMON_LOG" &
    PMON_PID=$!
}

stop_gpu_sampler () {
    [[ -n $GPU_SAMPLER_PID ]] && kill "$GPU_SAMPLER_PID"
    [[ $LOG_GPU -eq 0 ]] && return

    PEAK=$(awk -F',' 'NR>1 && $2>max{max=$2} END{print max}' "$RAW_LOG")
    echo "Peak GPU memory: ${PEAK} MiB"
    echo "job ${SLURM_JOB_ID}: peak ${PEAK} MiB" >> "$SUMMARY_LOG"
}

stop_proc_monitor () { [[ -n $PMON_PID ]] && kill $PMON_PID; }

export LD_LIBRARY_PATH=$HOME/Programs/MPCGPU/qdldl/build/out:$LD_LIBRARY_PATH

# Navigate to the directory containing your executables
cd $HOME/Programs/MPCGPU

# Move previous output logs except current one to backups
CURRENT_OUT="mpcgpu_${SLURM_JOB_ID}.out"
for file in mpcgpu_*.out; do
	if [[ "$file" != "$CURRENT_OUT" ]]; then
	    mv "$file" ./backups/
	fi
done

if [ -d "./results" ]; then
	rm -rf ./results
fi
mkdir -p ./results

# Execute your program
start_gpu_sampler
start_proc_monitor
python -u "$SCRIPT" "$@" || exit $?
stop_gpu_sampler
stop_proc_monitor

# Backup the new results
echo "Backing up results... in ./backups/${SLURM_JOB_ID}_results"
mkdir -p ./backups
cp -r ./results ./backups/${SLURM_JOB_ID}_results


# copy the current log into the result directory
cp "$CURRENT_OUT" ./results/
