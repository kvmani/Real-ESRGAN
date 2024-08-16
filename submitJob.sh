#!/bin/bash
#SBATCH --job-name=SR_inference
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00

# Function to log a message with a timestamp
log_message() {
  echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# Function to log system details
log_system_details() {
  log_message "System Details:"
  echo "----------------------------------------"
  echo "Hostname: $(hostname)"
  echo "Kernel: $(uname -r)"
  echo "OS: $(lsb_release -d -s 2>/dev/null || echo 'OS details not available')"
  echo "CPU: $(lscpu | grep 'Model name:' | awk -F ':' '{print $2}')"
  echo "Memory: $(free -h | grep Mem: | awk '{print $2}')"
  echo "----------------------------------------"
}

# Log system details
log_system_details

# Record start time
start_time=$(date +%s)

log_message "Script execution started."

# Check GPU availability
log_message "Checking GPU status with nvidia-smi..."
nvidia-smi

# Set the CUDA_VISIBLE_DEVICES environment variable to use a specific GPU
export CUDA_VISIBLE_DEVICES=0
log_message "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# Show network configuration
log_message "Checking network configuration with ifconfig..."
ifconfig

# Terminate any running Python processes to avoid conflicts
log_message "Terminating all running Python processes..."
killall -9 python3

# Display the current Python and pip paths before activating the environment
log_message "Python and pip paths before environment activation:"
which python3
which pip3

# Activate the Anaconda environment
log_message "Activating Anaconda environment..."
source $HOME/anaconda3/bin/activate
conda env list
conda activate ml_env

# Display the current Python and pip paths after activating the environment
log_message "Python and pip paths after environment activation:"
which python3
which pip3

# Log the current working directory
log_message "Current working directory:"
pwd


# Number of CPU cores to use (replace N with the desired number of cores)
N=48  # assigned from the command line argument

# Input and output directories
INPUT_DIR="./inference_testing/small_kikuchi_lr_8X8"
MODEL_PATH_1="experiments/train_RealESRNetx4plus_Kikuchi_4GPU/models/net_g_100000.pth"

# Log the configuration
log_message "Configuration:"
echo "Input Directory: $INPUT_DIR"
echo "Model Path: $MODEL_PATH_1"
echo "Number of Processes: $N"

# Make GPU invisible to the script
export CUDA_VISIBLE_DEVICES=""
log_message "Switched off the GPUs to ensure I run on CPUs only"



# Create temporary directories to split images
log_message "Creating temporary directories for image splitting..."
mkdir -p $INPUT_DIR/tmp

# Get the list of images
images=($(find $INPUT_DIR -maxdepth 1 -type f))
total_images=${#images[@]}
images_per_folder=$(( (total_images + N - 1) / N ))

# Split images into N folders
log_message "Splitting images into $N folders..."
for ((i=0; i<N; i++)); do
  folder="$INPUT_DIR/tmp/input_split_$i"
  mkdir -p "$folder"
   
  start=$((i * images_per_folder))
  end=$((start + images_per_folder))
  
  if [ $start -lt $total_images ]; then
    if [ $end -gt $total_images ]; then
      end=$total_images
    fi
    
    current_images=("${images[@]:start:end-start}")

    for img in "${current_images[@]}"; do
      cp -v "$img" "$folder/"
    done
  fi
done

log_message "Image splitting completed."

# Run the Python script for parallel processing
log_message "Starting parallel image processing..."
python3 multiprocessing_script.py --input_dir $INPUT_DIR --model_path $MODEL_PATH_1 --output_dir results/ExtraSR_RealSRNet_CPU_100000 --num_processes $N

log_message "Parallel image processing completed."

# Record end time and calculate total execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))

log_message "Script execution finished."
log_message "Total execution time: $execution_time seconds."
