#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <chrono>
#include <iostream>

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

// Function to create a phantom with a circle and two ellipses, with intensity varying from the outer edge to the center. This phantom is assumed to be the distributed SPIONs that detectable in body because of scanning on FFL
std::vector<std::vector<double>> create_phantom(int size) {
    std::vector<std::vector<double>> phantom(size, std::vector<double>(size, 0.0));
    std::vector<double> x(size), y(size);
    double step = 2.0 / (size - 1);
    
    for (int i = 0; i < size; ++i) {
        x[i] = -1 + i * step;
        y[i] = -1 + i * step;
    }

    // Circle with varying intensity
    double circle_radius = 0.1;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double distance_from_center = std::sqrt(x[i] * x[i] + y[j] * y[j]);
            if (distance_from_center <= circle_radius) {
                phantom[j][i] = 0.5 * (1.0 - distance_from_center / circle_radius);
            }
        }
    }

    // First ellipse with varying intensity
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double dx1 = (x[i] - 0.3) / 0.2;
            double dy1 = (y[j] - 0.3) / 0.1;
            double ellipse1_distance = std::sqrt(dx1 * dx1 + dy1 * dy1);
            if (ellipse1_distance <= 1.0) {
                phantom[j][i] = 1.0 * (1.0 - ellipse1_distance);
            }
        }
    }

    // Second ellipse with varying intensity
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double dx2 = (x[i] + 0.3) / 0.2;
            double dy2 = (y[j] + 0.3) / 0.1;
            double ellipse2_distance = std::sqrt(dx2 * dx2 + dy2 * dy2);
            if (ellipse2_distance <= 1.0) {
                phantom[j][i] = 1.0 * (1.0 - ellipse2_distance);
            }
        }
    }

    return phantom;
}

// Results conversion to txt file in order to be visualized using Python
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

// Sinogram function
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

    // Create phantom
    auto phantom = create_phantom(size);
    save_phantom_to_txt(phantom, "phantom.txt");

    // Compute Radon transform (sinogram)
    auto angles = linspace(0.0, 180.0, size);
    auto sinogram = radon_transform(phantom, angles);
    save_phantom_to_txt(sinogram, "sinogram.txt");

    // Perform ART with TV regularization
    auto reconstruction = algebraic_reconstruction_tv(sinogram, angles, iterations, alpha, beta, step_size);
    save_phantom_to_txt(reconstruction, "reconstruction_tv.txt");

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
