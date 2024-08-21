import cupy as cp
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import csv
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure CuPy is using the correct GPU
cp.cuda.runtime.setDevice(0)

def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)['Close']
    return stock_data

@cp.fuse()
def simulate_prices(prev_prices, random_returns):
    return prev_prices * (1 + random_returns)

def monte_carlo_simulation_gpu(start_price, days, num_simulations, vram_limit, csv_filename):
    logging.info(f"Starting Monte Carlo simulation with {num_simulations} simulations for {days} days")

    # Calculate how many simulations can fit in VRAM
    sim_size = days * 4  # 4 bytes per float32
    vram_sims = int(vram_limit / sim_size)
    
    logging.info(f"Running {vram_sims} simulations on GPU")

    try:
        with cp.cuda.Device(0):
            stream = cp.cuda.get_current_stream()
            with stream:
                # Preallocate memory on GPU
                d_prices = cp.full((days, vram_sims), start_price, dtype=cp.float32)
                
                # Create random states
                rng_gpu = cp.random.RandomState()

                # Open CSV file for writing simulation data progressively
                with open(csv_filename, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['Day'] + [f'Sim_{i+1}' for i in range(vram_sims)])

                    # Run simulation
                    for i in tqdm(range(1, days), desc="Simulating"):
                        # GPU simulation
                        random_returns_gpu = rng_gpu.normal(0, 0.02, size=vram_sims).astype(cp.float32)
                        d_prices[i] = simulate_prices(d_prices[i-1], random_returns_gpu)

                        # Synchronize after each iteration
                        stream.synchronize()

                        # Move data to CPU and save to CSV
                        row = [i] + d_prices[i].get().tolist()
                        csvwriter.writerow(row)

                # Calculate summary results
                gpu_results = cp.array([
                    cp.mean(d_prices, axis=1),
                    cp.median(d_prices, axis=1),
                    cp.percentile(d_prices, 5, axis=1),
                    cp.percentile(d_prices, 95, axis=1)
                ]).get()

        return gpu_results

    except cp.cuda.runtime.CUDARuntimeError as e:
        logging.error(f"CUDA error encountered: {e}")
        raise
    finally:
        # Release GPU resources
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

def main():
    ticker = input("Enter the stock ticker symbol: ")
    simulation_years = 25
    days_to_simulate = simulation_years * 252  # Assuming 252 trading days per year

    # Get available VRAM
    vram_info = cp.cuda.runtime.memGetInfo()
    available_vram = vram_info[0] * 0.9  # Use 90% of available VRAM

    # Get historical data for start and end prices
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    stock_data = get_stock_data(ticker, start_date, end_date)
    start_price = stock_data.iloc[0]

    logging.info(f"Starting price for {ticker}: {start_price}")
    logging.info("Running Monte Carlo simulation...")

    try:
        csv_filename = f"Monte_Carlo_Simulations_{days_to_simulate}_days.csv"
        results = monte_carlo_simulation_gpu(start_price, days_to_simulate, 0, available_vram, csv_filename)

        # Save summary results
        np.save(f"{ticker}_Monte_Carlo_Results_{simulation_years}years.npy", results)
        logging.info(f"Monte Carlo simulation summary results have been saved.")

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(results[:, 0], results[:, 1], label='Mean', color='blue', linewidth=2)
        plt.plot(results[:, 0], results[:, 2], label='Median', color='green', linewidth=2)
        plt.fill_between(results[:, 0], results[:, 3], results[:, 4], alpha=0.2, color='red', label='90% Confidence Interval')
        plt.title(f'{ticker} - Monte Carlo Simulation ({simulation_years} years)')
        plt.xlabel('Trading Days')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(f"{ticker}_Monte_Carlo_Simulation_{simulation_years}years.png")
        plt.show()

        logging.info(f"Graph saved as {ticker}_Monte_Carlo_Simulation_{simulation_years}years.png")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        logging.info("GPU Info:")
        logging.info(cp.cuda.runtime.getDeviceProperties(0))

if __name__ == "__main__":
    main()
