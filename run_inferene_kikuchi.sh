#!/bin/bash

echo "Super resolution inference - Start "
nvidia-smi
#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0
ifconfig

killall -9 python3

which python3
which pip3

echo above is before activation of env

source $HOME/anaconda3/bin/activate
conda env list
conda activate ml_env

#export JAVA_HOME="/home/lus04/$USER/jdk-21.0.1"
#export PATH="/home/lus04/$USER/jdk-21.0.1/bin:$PATH"
#export PATH="/home/lus04/$USER/anaconda-3.9.12/bin:$PATH"
echo below is after activation of env
which python3
which pip3

pwd

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

# Your long-running script logic goes here
log_message "Running inference with model at 100000 steps..."
python inference_realesrgan.py --input ./inference_testing/kikuchi_lr_1X1 --model_name RealESRNet_x4plus --output results/ExtraSR_RealSRNet_100000 --outscale 4 --model_path experiments/train_RealESRNetx4plus_Kikuchi_4GPU/models/net_g_100000.pth

log_message "First inference completed."

log_message "Running inference with model at 200000 steps..."
python inference_realesrgan.py --input ./inference_testing/kikuchi_lr_1X1 --model_name RealESRNet_x4plus --output results/Extra_SR_RealSRNet_220000 --outscale 4 --model_path experiments/train_RealESRNetx4plus_Kikuchi_4GPU/models/net_g_200000.pth

log_message "Second inference completed."

# Record the end time
END_TIME=$(date +%s)

# Calculate the total execution time
TOTAL_TIME=$(($END_TIME - $START_TIME))

# Log the end message and total execution time
log_message "Script execution finished."
log_message "Total execution time: $(($TOTAL_TIME / 60)) minutes and $(($TOTAL_TIME % 60)) seconds."


echo "Kikuchi Super resolution - Inferecne End"
