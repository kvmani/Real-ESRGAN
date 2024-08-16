import os
import argparse
import multiprocessing
from subprocess import call
from datetime import datetime

# Function to process images in a folder
def process_images(folder, model_path, output_folder):
    # Get the process ID and CPU ID
    process_id = os.getpid()
    cpu_id = os.sched_getaffinity(0)

    # Log the start of the process
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{start_time}] Process ID: {process_id}, CPU: {cpu_id} - Processing images in folder {folder}...")

    call(["python", "inference_realesrgan.py", "--input", folder, "--model_name", "RealESRNet_x4plus", "--output", output_folder, "--outscale", "4", "--model_path", model_path, "--fp32"])

    # Log the completion of the process
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{end_time}] Process ID: {process_id}, CPU: {cpu_id} - Processing completed for folder {folder}.")

# Prepare tasks for multiprocessing
def prepare_tasks(input_dir, model_path, output_dir):
    tasks = []
    for folder in os.listdir(os.path.join(input_dir, 'tmp')):
        tmp_folder = os.path.join(input_dir, 'tmp', folder)
        if os.path.isdir(tmp_folder):
            tasks.append((tmp_folder, model_path, output_dir))
    return tasks

# Run tasks in parallel using multiprocessing
def run_parallel_tasks(tasks, n_processes):
    with multiprocessing.Pool(n_processes) as pool:
        pool.starmap(process_images, tasks)

def main():
    parser = argparse.ArgumentParser(description="Parallel image processing")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--model_path', type=str, required=True, help='Model path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of parallel processes')

    args = parser.parse_args()

    # Prepare and run tasks
    tasks = prepare_tasks(args.input_dir, args.model_path, args.output_dir)
    run_parallel_tasks(tasks, args.num_processes)

if __name__ == "__main__":
    main()
