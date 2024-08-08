#!/bin/bash

echo "Super resolution - Start "
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

# Train pretrain
#python3 -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2_fashion/train/pretrain_stage2-base.yaml

#python ../test_gpu.py
python inference_realesrgan.py --input ./inference_testing  --model_name RealESRNet_x4plus --output results/RealSRNet_101000 --outscale 4 --model_path experiments/train_RealESRNetx4plus_1000k_mani_2nd/models/net_g_101000.pth

echo "completed one of the inferences"


python inference_realesrgan.py --input ./inference_testing  --model_name RealESRGAN_x4plus --output results/GAN_135000 --outscale 4 --model_path experiments/train_RealESRGANx4plus_400k_B12G4_mani_archived_20240609_221608/models/net_g_135000.pth

python inference_realesrgan.py --input ./inference_testing  --model_name RealESRGAN_x4plus --output results/GAN_230000 --outscale 4 --model_path experiments/train_RealESRGANx4plus_4GPU_trial/models/net_g_230000.pth


echo "completed one of the inferences"

# Evaluate pretrain
#python3 -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2_fashion/eval/caption_facad_opt2.7b_eval.yaml

# Train finetune.
#python3 -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2_fashion/train/finetune-base.yaml

# Evaluate finetune.
#python3 -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2_fashion/eval/caption_facad_opt2.7b_eval.yaml

echo "Super resolution - End"
