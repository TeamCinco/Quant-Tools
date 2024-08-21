import cupy as cp
import numpy as np
import pandas as pd
import psutil
import logging
from tqdm import tqdm
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def limit_memory_usage(target_ram_gb=63.8, target_gpu_vram_gb=10.8):
    total_ram = psutil.virtual_memory().total
    target_ram_bytes = int(target_ram_gb * 1e9)
    if target_ram_bytes > total_ram:
        logging.warning(f"Target RAM ({target_ram_gb:.1f} GB) exceeds system RAM ({total_ram / 1e9:.1f} GB). Using 90% of available RAM.")
        target_ram_bytes = int(0.9 * total_ram)
    
    try:
        cp.cuda.runtime.setDevice(0)
        free_gpu_memory, total_gpu_memory = cp.cuda.runtime.memGetInfo()
        target_gpu_vram_bytes = min(int(target_gpu_vram_gb * 1e9), int(0.9 * free_gpu_memory))
        logging.info(f"GPU acceleration enabled. Using {target_gpu_vram_bytes / 1e9:.1f} GB of GPU memory.")
    except Exception as e:
        logging.warning(f"Unable to set up GPU acceleration. Continuing without GPU support. Error: {str(e)}")
        target_gpu_vram_bytes = 0
    
    return target_ram_bytes, target_gpu_vram_bytes

@cp.fuse()
def calculate_price_differences(prices):
    daily_diff = cp.diff(prices, axis=0)
    weekly_diff = cp.diff(prices, periods=5, axis=0)
    monthly_diff = cp.diff(prices, periods=21, axis=0)
    return daily_diff, weekly_diff, monthly_diff

def calculate_std(differences):
    return cp.std(differences, axis=0)

def filter_paths_within_std(data, differences, std, std_multipliers=[1, 2, 3]):
    filtered_paths = {}
    for multiplier in std_multipliers:
        lower_bound = -multiplier * std
        upper_bound = multiplier * std
        mask = (differences >= lower_bound) & (differences <= upper_bound)
        filtered_paths[multiplier] = data[:, mask.any(axis=0)]
    return filtered_paths

def process_data(input_filename, output_filename, max_paths=10000, chunk_size=1000):
    with cp.cuda.Stream() as stream:
        data = pd.read_csv(input_filename, index_col='Day')
        days, num_sims = data.shape
        
        d_data = cp.asarray(data.values, dtype=cp.float32)
        
        daily_diff, weekly_diff, monthly_diff = calculate_price_differences(d_data)
        
        daily_std = calculate_std(daily_diff)
        weekly_std = calculate_std(weekly_diff)
        monthly_std = calculate_std(monthly_diff)
        
        daily_filtered = filter_paths_within_std(d_data, daily_diff, daily_std)
        weekly_filtered = filter_paths_within_std(d_data, weekly_diff, weekly_std)
        monthly_filtered = filter_paths_within_std(d_data, monthly_diff, monthly_std)
        
        all_filtered_paths = cp.concatenate([daily_filtered[1], daily_filtered[2], daily_filtered[3],
                                             weekly_filtered[1], weekly_filtered[2], weekly_filtered[3],
                                             monthly_filtered[1], monthly_filtered[2], monthly_filtered[3]], axis=1)
        
        all_filtered_paths = all_filtered_paths[:, :min(max_paths, all_filtered_paths.shape[1])]
        
        df_filtered = pd.DataFrame(all_filtered_paths.get(), index=data.index)
        df_filtered.to_csv(output_filename)
        
        return daily_std.get(), weekly_std.get(), monthly_std.get()

def main():
    target_ram, target_gpu_vram = limit_memory_usage()
    logging.info(f"Target RAM usage: {target_ram / 1e9:.1f} GB")
    logging.info(f"Target GPU VRAM usage: {target_gpu_vram / 1e9:.1f} GB")
    
    input_csv = r"C:\Users\cinco\Desktop\quant practicie\spy_Monte_Carlo_Simulation_2322_days_1000000_sims.csv"
    output_csv = r'C:\Users\cinco\Desktop\quant practicie\filtered_paths.csv'
    
    try:
        daily_std, weekly_std, monthly_std = process_data(input_csv, output_csv)
        
        logging.info(f"Daily STD: {daily_std}")
        logging.info(f"Weekly STD: {weekly_std}")
        logging.info(f"Monthly STD: {monthly_std}")
        logging.info(f"Filtered paths saved to: {output_csv}")
    except Exception as e:
        logging.error(f"An error occurred during data processing: {str(e)}")
    finally:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

if __name__ == "__main__":
    main()