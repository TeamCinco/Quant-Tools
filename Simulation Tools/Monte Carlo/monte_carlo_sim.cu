#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

// CUDA kernel for calculating price differences
__global__ void calculatePriceDifferences(float* prices, float* daily_diff, float* weekly_diff, float* monthly_diff, int days, int num_sims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sims) {
        for (int i = 1; i < days; ++i) {
            daily_diff[(i-1)*num_sims + idx] = prices[i*num_sims + idx] - prices[(i-1)*num_sims + idx];
            if (i >= 5) {
                weekly_diff[(i-5)*num_sims + idx] = prices[i*num_sims + idx] - prices[(i-5)*num_sims + idx];
            }
            if (i >= 21) {
                monthly_diff[(i-21)*num_sims + idx] = prices[i*num_sims + idx] - prices[(i-21)*num_sims + idx];
            }
        }
    }
}

// Function to calculate standard deviation
float calculateStd(thrust::device_vector<float>& data) {
    float mean = thrust::reduce(data.begin(), data.end(), 0.0f, thrust::plus<float>()) / data.size();
    thrust::transform(data.begin(), data.end(), data.begin(), [mean] __device__ (float x) { return (x - mean) * (x - mean); });
    float variance = thrust::reduce(data.begin(), data.end(), 0.0f, thrust::plus<float>()) / (data.size() - 1);
    return std::sqrt(variance);
}

// Function to filter paths within standard deviation
thrust::device_vector<float> filterPathsWithinStd(thrust::device_vector<float>& data, thrust::device_vector<float>& differences, float std, float multiplier) {
    thrust::device_vector<bool> mask(differences.size());
    float lower_bound = -multiplier * std;
    float upper_bound = multiplier * std;
    thrust::transform(differences.begin(), differences.end(), mask.begin(),
                      [=] __device__ (float x) { return x >= lower_bound && x <= upper_bound; });
    
    thrust::device_vector<float> filtered_paths(data.size());
    auto new_end = thrust::copy_if(thrust::device, data.begin(), data.end(), mask.begin(), filtered_paths.begin());
    filtered_paths.resize(thrust::distance(filtered_paths.begin(), new_end));
    return filtered_paths;
}

int main() {
    // Read input CSV
    std::string input_csv = R"(C:\Users\cinco\Desktop\quant practicie\spy_Monte_Carlo_Simulation_2322_days_1000000_sims.csv)";
    std::string output_csv = R"(C:\Users\cinco\Desktop\quant practicie\filtered_paths.csv)";

    std::ifstream file(input_csv);
    std::string line;
    std::vector<std::vector<float>> data;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<float> row;
        std::istringstream iss(line);
        std::string value;
        
        // Skip the first column (Day)
        std::getline(iss, value, ',');

        while (std::getline(iss, value, ',')) {
            row.push_back(std::stof(value));
        }
        data.push_back(row);
    }

    int days = data.size();
    int num_sims = data[0].size();

    // Copy data to GPU
    thrust::device_vector<float> d_prices(days * num_sims);
    for (int i = 0; i < days; ++i) {
        thrust::copy(data[i].begin(), data[i].end(), d_prices.begin() + i * num_sims);
    }

    // Calculate price differences
    thrust::device_vector<float> d_daily_diff((days-1) * num_sims);
    thrust::device_vector<float> d_weekly_diff((days-5) * num_sims);
    thrust::device_vector<float> d_monthly_diff((days-21) * num_sims);

    int threadsPerBlock = 256;
    int blocks = (num_sims + threadsPerBlock - 1) / threadsPerBlock;
    calculatePriceDifferences<<<blocks, threadsPerBlock>>>(
        thrust::raw_pointer_cast(d_prices.data()),
        thrust::raw_pointer_cast(d_daily_diff.data()),
        thrust::raw_pointer_cast(d_weekly_diff.data()),
        thrust::raw_pointer_cast(d_monthly_diff.data()),
        days, num_sims
    );

    // Calculate standard deviations
    float daily_std = calculateStd(d_daily_diff);
    float weekly_std = calculateStd(d_weekly_diff);
    float monthly_std = calculateStd(d_monthly_diff);

    std::cout << "Daily STD: " << daily_std << std::endl;
    std::cout << "Weekly STD: " << weekly_std << std::endl;
    std::cout << "Monthly STD: " << monthly_std << std::endl;

    // Filter paths
    std::vector<float> std_multipliers = {1, 2, 3};
    std::vector<thrust::device_vector<float>> filtered_paths;

    for (float multiplier : std_multipliers) {
        filtered_paths.push_back(filterPathsWithinStd(d_prices, d_daily_diff, daily_std, multiplier));
        filtered_paths.push_back(filterPathsWithinStd(d_prices, d_weekly_diff, weekly_std, multiplier));
        filtered_paths.push_back(filterPathsWithinStd(d_prices, d_monthly_diff, monthly_std, multiplier));
    }

    // Combine filtered paths
    thrust::device_vector<float> all_filtered_paths;
    for (const auto& paths : filtered_paths) {
        all_filtered_paths.insert(all_filtered_paths.end(), paths.begin(), paths.end());
    }

    // Limit to max_paths
    int max_paths = 10000;
    if (all_filtered_paths.size() > max_paths * days) {
        all_filtered_paths.resize(max_paths * days);
    }

    // Save filtered paths to CSV
    thrust::host_vector<float> h_filtered_paths = all_filtered_paths;
    std::ofstream outfile(output_csv);
    outfile << "Day";
    for (int i = 0; i < h_filtered_paths.size() / days; ++i) {
        outfile << ",Path_" << i;
    }
    outfile << std::endl;

    for (int i = 0; i < days; ++i) {
        outfile << i;
        for (int j = 0; j < h_filtered_paths.size() / days; ++j) {
            outfile << "," << h_filtered_paths[i + j * days];
        }
        outfile << std::endl;
    }

    std::cout << "Filtered paths saved to: " << output_csv << std::endl;

    return 0;
}