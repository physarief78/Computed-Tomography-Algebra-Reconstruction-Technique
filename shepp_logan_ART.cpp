#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <chrono>
#include <string>
#include <sstream>

const double PI = 3.14159265358979323846;

// Helper function to generate linearly spaced values (similar to Python's linspace)
std::vector<double> linspace(double start, double end, int num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

// Function to load the Shepp-Logan phantom from a text file
std::vector<std::vector<double>> load_phantom_from_file(const std::string& filename) {
    std::vector<std::vector<double>> phantom;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for reading." << std::endl;
        return phantom;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        double value;
        while (ss >> value) {
            row.push_back(value);
        }
        phantom.push_back(row);
    }

    file.close();
    std::cout << "Phantom loaded from " << filename << std::endl;
    return phantom;
}

void save_phantom_to_txt(const std::vector<std::vector<double>>& phantom, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    int size = phantom.size();
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            file << phantom[i][j] << " ";
        }
        file << "\n"; 
    }
    
    file.close();
    std::cout << "Phantom image saved to " << filename << std::endl;
}

std::vector<std::vector<double>> radon_transform(const std::vector<std::vector<double>>& image, const std::vector<double>& angles) {
    int size = image.size();
    int center = size / 2;
    std::vector<std::vector<double>> sinogram(angles.size(), std::vector<double>(size, 0.0));

    for (size_t i = 0; i < angles.size(); i++) {
        double theta = angles[i] * PI / 180.0;
        for (int t = -center; t < center; t++) {
            double sum_val = 0.0;
            for (int s = -center; s < center; s++) {
                int x = static_cast<int>(center + t * cos(theta) - s * sin(theta));
                int y = static_cast<int>(center + t * sin(theta) + s * cos(theta));
                if (x >= 0 && x < size && y >= 0 && y < size) {
                    sum_val += image[y][x];
                }
            }
            sinogram[i][t + center] = sum_val;
        }
    }
    return sinogram;
}

// Compute the TV gradient
std::vector<std::vector<double>> tv_gradient(const std::vector<std::vector<double>>& image) {
    int size = image.size();
    std::vector<std::vector<double>> grad(size, std::vector<double>(size, 0.0));

    for (int i = 0; i < size - 1; ++i) {
        for (int j = 0; j < size - 1; ++j) {
            double grad_x = image[i + 1][j] - image[i][j];
            double grad_y = image[i][j + 1] - image[i][j];
            grad[i][j] = std::sqrt(grad_x * grad_x + grad_y * grad_y);
        }
    }
    return grad;
}

// ART with TV regularization
std::vector<std::vector<double>> algebraic_reconstruction_tv(
    const std::vector<std::vector<double>>& sinogram, 
    const std::vector<double>& angles, 
    int iterations, 
    double alpha, 
    double beta, 
    double step_size) 
{
    int size = sinogram[0].size();
    int center = size / 2;
    std::vector<std::vector<double>> reconstruction(size, std::vector<double>(size, 0.0));

    for (int iter = 0; iter < iterations; iter++) {
        for (size_t i = 0; i < angles.size(); i++) {
            double theta = angles[i] * PI / 180.0;
            for (int t = -center; t < center; t++) {
                double sum_val = 0.0;
                int count = 0;
                for (int s = -center; s < center; s++) {
                    int x = static_cast<int>(center + t * cos(theta) - s * sin(theta));
                    int y = static_cast<int>(center + t * sin(theta) + s * cos(theta));
                    if (x >= 0 && x < size && y >= 0 && y < size) {
                        sum_val += reconstruction[y][x];
                        count++;
                    }
                }
                if (count > 0) {
                    double correction = (sinogram[i][t + center] - sum_val) / count;
                    for (int s = -center; s < center; s++) {
                        int x = static_cast<int>(center + t * cos(theta) - s * sin(theta));
                        int y = static_cast<int>(center + t * sin(theta) + s * cos(theta));
                        if (x >= 0 && x < size && y >= 0 && y < size) {
                            reconstruction[y][x] += correction;
                        }
                    }
                }
            }
        }

        // TV regularization gradient descent
        auto grad_tv = tv_gradient(reconstruction);
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                reconstruction[i][j] -= step_size * (alpha * reconstruction[i][j] + beta * grad_tv[i][j]);
            }
        }
    }

    return reconstruction;
}

// Compute MSE
double compute_mse(const std::vector<std::vector<double>>& original, const std::vector<std::vector<double>>& reconstructed) {
    int size = original.size();
    double mse = 0.0;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            mse += pow(original[i][j] - reconstructed[i][j], 2);
        }
    }
    return mse / (size * size);
}

// Compute PSNR
double compute_psnr(const std::vector<std::vector<double>>& original, const std::vector<std::vector<double>>& reconstructed) {
    double mse = compute_mse(original, reconstructed);
    double max_pixel_value = 1.0;
    if (mse > 0) {
        return 10 * log10(max_pixel_value * max_pixel_value / mse);
    }
    return std::numeric_limits<double>::infinity();
}

int main() {
    // Measure start time
    auto start = std::chrono::high_resolution_clock::now();

    // Parameters
    int size = 512;
    int iterations = 20;
    double alpha = 1.0;
    double beta = 0.01;
    double step_size = 0.1;
    std::string phantom_filename = "shepp_logan_phantom.txt";  // Phantom file generated from Python

     // Load the Shepp-Logan phantom from file
    auto phantom = load_phantom_from_file(phantom_filename);
    if (phantom.empty()) {
        std::cerr << "Failed to load the phantom." << std::endl;
        return -1;
    }

    // Compute Radon transform (sinogram)
    auto angles = linspace(0.0, 180.0, size);
    auto sinogram = radon_transform(phantom, angles);
    save_phantom_to_txt(sinogram, "sinogram_sl.txt");

    // Perform ART with TV regularization
    auto reconstruction = algebraic_reconstruction_tv(sinogram, angles, iterations, alpha, beta, step_size);
    save_phantom_to_txt(reconstruction, "reconstruction_tv_sl.txt");

    // Compute quality metrics
    double mse = compute_mse(phantom, reconstruction);
    double psnr = compute_psnr(phantom, reconstruction);

    std::cout << "\nMSE: " << mse << "\n";
    std::cout << "PSNR: " << psnr << " dB\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution Time: " << elapsed.count() << " seconds\n";

    return 0;
}









