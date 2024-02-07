#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=15:0:0    
#SBATCH --mail-user=<youremail@gmail.com>
#SBATCH --mail-type=ALL

#!/bin/bash
#SBATCH --job-name=deep_learning_job   # Job name
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task        
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=32G                      # Job memory request
#SBATCH --time=02:00:00                # Time limit hrs:min:sec
#SBATCH --output=/home/kka151/projects/def-t55wang/kka151/wfd-gan/deep_learning_job_%j.log  # Standard output and error log
#SBATCH --gres=gpu:1                   # Request GPU resource

# Load any modules and activate your conda environment here
module load python/3.10.2
module load cuda/11.7
source /home/kka151/venvs/torch/bin/activate


# Navigate to your project directory (optional)
cd /home/kka151/projects/def-t55wang/kka151/wfd-gan/src/



# Execute your deep learning script
python3 extract.py --dir /home/kka151/scratch/Tor-DS/ds19 --format .cell --length 2000
