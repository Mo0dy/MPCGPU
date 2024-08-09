#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cstdio>
#include <cuda_runtime.h>

struct MPCLogParams {
    int start_state_index;
    int goal_state_index;
    uint32_t test_iter;
    std::string test_output_prefix;
};

template <typename T>
void dump_tracking_data(std::vector<int> *pcg_iters, 
                        std::vector<bool> *pcg_exits, 
                        std::vector<double> *linsys_times, 
                        std::vector<double> *sqp_times, 
                        std::vector<uint32_t> *sqp_iters, 
                        std::vector<bool> *sqp_exits, 
                        std::vector<T> *tracking_errors, 
                        std::vector<std::vector<T>> *tracking_path, 
                        uint32_t timesteps_taken, 
                        uint32_t control_updates_taken, 
                        MPCLogParams mpc_log_params){
    // Helper function to create file names
    auto createFileName = [&](const std::string& data_type) {
        std::string filename = mpc_log_params.test_output_prefix + "_" + std::to_string(mpc_log_params.test_iter) + "_" + data_type + ".result";
        return filename;
    };
    
    // Helper function to dump single-dimension vector data
    auto dumpVectorData = [&](const auto& data, const std::string& data_type) {
        std::ofstream file(createFileName(data_type));
        if (!file.is_open()) {
            std::cerr << "Failed to open " << data_type << " file.\n";
            return;
        }
        for (const auto& item : *data) {
            file << item << '\n';
        }
        file.close();
    };

    // Dump single-dimension vector data
    dumpVectorData(pcg_iters, "pcg_iters");
    dumpVectorData(linsys_times, "linsys_times");
    dumpVectorData(sqp_times, "sqp_times");
    dumpVectorData(sqp_iters, "sqp_iters");
    dumpVectorData(sqp_exits, "sqp_exits");
    dumpVectorData(tracking_errors, "tracking_errors");
    dumpVectorData(pcg_exits, "pcg_exits");


    // Dump two-dimension vector data (tracking_path)
    std::ofstream file(createFileName("tracking_path"));
    if (!file.is_open()) {
        std::cerr << "Failed to open tracking_path file.\n";
        return;
    }
    for (const auto& outerItem : *tracking_path) {
        for (const auto& innerItem : outerItem) {
            file << innerItem << ',';
        }
        file << '\n';
    }
    file.close();

    std::ofstream statsfile(createFileName("stats"));
    if (!statsfile.is_open()) {
        std::cerr << "Failed to open stats file.\n";
        return;
    }
    statsfile << "timesteps: " << timesteps_taken << "\n";
    statsfile << "control_updates: " << control_updates_taken << "\n";
    // printStatsToFile<double>(&linsys_times, )
    
    statsfile.close();
}

// read a CSV file into a vector of vectors
template <typename T>
std::vector<std::vector<T>> readCSVToVecVec(const std::string& filename) {
    std::vector<std::vector<T>> data;
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        std::cerr << "File [ " << filename << " ] could not be opened!\n";
    } else {
        std::string line;


        while (std::getline(infile, line)) {
            std::vector<T> row;
            std::stringstream ss(line);
            std::string val;

            while (std::getline(ss, val, ',')) {
                row.push_back(std::stof(val));
            }

            data.push_back(row);
        }
    }

    infile.close();
    return data;
}

// read a CSV file into a vector
template <typename T>
std::vector<T> readCSVToVec(const std::string& filename) {
    std::vector<T> data;
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        std::cerr << "File [ " << filename << " ] could not be opened!\n";
    } else {
        std::string line;

        while (std::getline(infile, line)) {
            std::stringstream ss(line);
            std::string val;

            while (std::getline(ss, val, ',')) {
                data.push_back(static_cast<T>(std::stof(val)));
            }
        }
    }

    infile.close();
    return data;
}

// Format stats string values into CSV format
std::string getStatsString(const std::string& statsString) {
    std::stringstream ss(statsString);
    std::string token;
    std::string csvFormattedString;
    
    while (getline(ss, token, '[')) {
        if (getline(ss, token, ']')) {
            if (!csvFormattedString.empty()) {
                csvFormattedString += ",";
            }
            csvFormattedString += token;
        }
    }
    
    return csvFormattedString;
 }

int writeResultsToCSV(const std::string& filename, const std::string& trackingStats, const std::string& linsysOrSqpStats){
   // Open the CSV file for writing
   std::ofstream csvFile(filename);
   if (!csvFile.is_open()) {
       std::cerr << "Error opening CSV file for writing." << std::endl;
       return 1;
   }

   // Write the header row
   csvFile << "Average, Std Dev, Min, Max, Median, Q1, Q3\n";
   // Write the data rows
   csvFile << getStatsString(trackingStats) << "\n";
   csvFile << getStatsString(linsysOrSqpStats) << "\n";

   // Close the CSV file
   csvFile.close();

   return 0;
}

void write_device_matrix_to_file(float* d_matrix, int rows, int cols, const char* filename, int filesuffix = 0) {
    
    char fname[100];
    snprintf(fname, sizeof(fname), "%s%d.txt", filename, filesuffix);
    
    // Allocate host memory for the matrix
    float* h_matrix = new float[rows * cols];

    // Copy the data from the device to the host memory
    size_t pitch = cols * sizeof(float);
    cudaMemcpy2D(h_matrix, pitch, d_matrix, pitch, pitch, rows, cudaMemcpyDeviceToHost);

    // Write the data to a file in column-major order
    std::ofstream outfile(fname);
    if (outfile.is_open()) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                outfile << std::setprecision(std::numeric_limits<float>::max_digits10+1) << h_matrix[col * rows + row] << "\t";
            }
            outfile << std::endl;
        }
        outfile.close();
    } else {
        std::cerr << "Unable to open file: " << fname << std::endl;
    }

    // Deallocate host memory
    delete[] h_matrix;
}