#!/bin/bash

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

# Record the start time
START_TIME=$(date +%s)

# Log the start message and system details
log_message "Script execution started."
log_system_details

# Make GPU invisible to the script
log_message "Disabling GPU..."
export CUDA_VISIBLE_DEVICES=""

# Number of CPU cores to use (replace N with the desired number of cores)
N=6  # Change this value as per your requirement

# Input and output directories
INPUT_DIR="./inference_testing/kikuchi_lr_8X8"
MODEL_PATH_1="experiments/train_RealESRNetx4plus_Kikuchi_4GPU/models/net_g_100000.pth"
MODEL_PATH_2="experiments/train_RealESRNetx4plus_Kikuchi_4GPU/models/net_g_200000.pth"

# Create temporary directories to split images
log_message "Splitting input images into $N temporary folders..."
mkdir -p $INPUT_DIR/tmp

# Split the images into N parts

log_message "Splitting input images into $N temporary folders..."
mkdir -p $INPUT_DIR/tmp

# Get the list of images
images=($(find $INPUT_DIR -maxdepth 1 -type f))
total_images=${#images[@]}
images_per_folder=$((total_images / N + 1))

# Split images into N folders
for ((i=0; i<N; i++)); do
  folder="$INPUT_DIR/tmp/input_split_$i"
  mkdir -p "$folder"
  
  start=$((i * images_per_folder))
  end=$((start + images_per_folder))
  if [ $end -gt $total_images ]; then
    end=$total_images
  fi
  
  # Move images into the new folder
  mv "${images[@]:start:end-start}" "$folder/"
done


#find $INPUT_DIR -maxdepth 1 -type f | split -d -l $(($(find $INPUT_DIR -maxdepth 1 -type f | wc -l) / N + 1)) - $INPUT_DIR/tmp/input_split_

# Python script to process images using multiprocessing
log_message "Running inference using multiprocessing..."

python3 << EOF
import os
import multiprocessing
from subprocess import call

# Function to process images in a folder
def process_images(folder, model_path, output_folder):
    print(f"Processing images in folder {folder}...")
    call(["python", "inference_realesrgan.py", "--input", folder, "--model_name", "RealESRNet_x4plus", "--output", output_folder, "--outscale", "4", "--model_path", model_path, "--fp32"])
    print(f"Processing completed for folder {folder}.")

# Prepare tasks for multiprocessing
def prepare_tasks(input_dir, model_path, output_dir):
    print(f"{input_dir}, {model_path}, {output_dir}")
    tasks = []
    for folder in os.listdir(input_dir + '/tmp'):
        tmp_folder = os.path.join(input_dir, 'tmp', folder)
        if os.path.isdir(tmp_folder):
            #output_folder = os.path.join(output_dir, folder)
            tasks.append((tmp_folder, model_path, output_dir))
			
    return tasks

# Run tasks in parallel using multiprocessing
def run_parallel_tasks(tasks, n_processes):
    with multiprocessing.Pool(n_processes) as pool:
        pool.starmap(process_images, tasks)

# Main execution
if __name__ == "__main__":
    input_dir = "$INPUT_DIR"
    model_path_1 = "$MODEL_PATH_1"
    model_path_2 = "$MODEL_PATH_2"
    output_dir_1 = "results/ExtraSR_RealSRNet_100000"
    output_dir_2 = "results/Extra_SR_RealSRNet_220000"

    # Tasks for first inference
    tasks = prepare_tasks(input_dir, model_path_1, output_dir_1)
    print(tasks)
    run_parallel_tasks(tasks, $N)

    print("First inference completed.")

    # Tasks for second inference
    tasks = prepare_tasks(input_dir, model_path_2, output_dir_2)
    run_parallel_tasks(tasks, $N)

    print("Second inference completed.")
EOF

# Clean up temporary directories
log_message "Cleaning up temporary directories..."
rm -rf $INPUT_DIR/tmp

# Record the end time
END_TIME=$(date +%s)

# Calculate the total execution time
TOTAL_TIME=$(($END_TIME - $START_TIME))

# Log the end message and total execution time
log_message "Script execution finished."
log_message "Total execution time: $(($TOTAL_TIME / 60)) minutes and $(($TOTAL_TIME % 60)) seconds."

echo "Kikuchi Super resolution - Inference End"
