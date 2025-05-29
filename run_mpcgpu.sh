#!/bin/bash
#SBATCH --job-name=mpcgpu_job
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=mpcgpu_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=felix.muehlenberend@mailbox.org

# Load necessary modules (if required)
module load devel/cuda

export LD_LIBRARY_PATH=$HOME/Programs/MPCGPU/qdldl/build/out:$LD_LIBRARY_PATH

# Navigate to the directory containing your executables
cd $HOME/Programs/MPCGPU

# Execute your programs
python runner.py
# ./examples/pcg.exe
# ./examples/qdldl.exe
