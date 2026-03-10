// multithreaded Monte Carlo pricer for rainbow option: (max(S1,...,Sn) - K)+
// correlated GBM via Cholesky, OpenMP parallelism
// compile: g++ -O3 -std=c++17 -fopenmp mc_rainbow.cpp -o mc_rainbow -lm

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <thread>
#include <iomanip>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

Mat cholesky(const Mat& A) { // lower triangular L such that A = L*L^T
    int n = A.size();
    Mat L(n, Vec(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) sum += L[i][k] * L[j][k];
            if (i == j) L[i][j] = std::sqrt(A[i][i] - sum);
            else L[i][j] = (A[i][j] - sum) / L[j][j];
        }
    }
    return L;
}

struct MCResult {
    double price;
    double stderr;
    double elapsed_ms;
};

MCResult mc_rainbow(const Vec& S0, double K, double T, double r,
                    const Vec& sigmas, const Mat& corr, int n_paths, int n_threads) {
    int d = S0.size();
    Mat L = cholesky(corr);

    // precompute drifts
    Vec drift(d);
    for (int i = 0; i < d; i++)
        drift[i] = (r - 0.5 * sigmas[i] * sigmas[i]) * T;

    double sqrt_T = std::sqrt(T);
    double discount = std::exp(-r * T);

    // per-thread accumulators
    std::vector<double> thread_sum(n_threads, 0.0);
    std::vector<double> thread_sum2(n_threads, 0.0);
    int paths_per_thread = n_paths / n_threads;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; t++) {
        threads.emplace_back([&, t]() {
            std::mt19937_64 rng(42 + t); // different seed per thread
            std::normal_distribution<double> norm(0.0, 1.0);

            Vec Z(d), W(d), S_T(d);
            double local_sum = 0.0, local_sum2 = 0.0;

            for (int p = 0; p < paths_per_thread; p++) {
                // generate independent normals
                for (int i = 0; i < d; i++) Z[i] = norm(rng);

                // correlate via Cholesky: W = L * Z
                for (int i = 0; i < d; i++) {
                    W[i] = 0.0;
                    for (int j = 0; j <= i; j++) W[i] += L[i][j] * Z[j];
                }

                // terminal stock prices
                double max_S = 0.0;
                for (int i = 0; i < d; i++) {
                    S_T[i] = S0[i] * std::exp(drift[i] + sigmas[i] * sqrt_T * W[i]);
                    max_S = std::max(max_S, S_T[i]);
                }

                double payoff = std::max(max_S - K, 0.0);
                local_sum += payoff;
                local_sum2 += payoff * payoff;
            }

            thread_sum[t] = local_sum;
            thread_sum2[t] = local_sum2;
        });
    }

    for (auto& th : threads) th.join();

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    int total_paths = paths_per_thread * n_threads;
    double sum = 0.0, sum2 = 0.0;
    for (int t = 0; t < n_threads; t++) {
        sum += thread_sum[t];
        sum2 += thread_sum2[t];
    }

    double mean_payoff = sum / total_paths;
    double var = sum2 / total_paths - mean_payoff * mean_payoff;
    double price = discount * mean_payoff;
    double stderr = discount * std::sqrt(var / total_paths);

    return {price, stderr, elapsed};
}

int main() {
    int n_threads = std::thread::hardware_concurrency();
    std::cout << "Threads: " << n_threads << "\n\n";

    // --- 2-asset ---
    {
        Vec S0 = {100, 100};
        Vec sigmas = {0.2, 0.3};
        Mat corr = {{1.0, 0.5}, {0.5, 1.0}};

        std::cout << "=== 2-ASSET RAINBOW (10M paths) ===\n";
        auto r = mc_rainbow(S0, 100, 1.0, 0.05, sigmas, corr, 10'000'000, n_threads);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Price:  " << r.price << " ± " << r.stderr << "\n";
        std::cout << "Time:   " << std::setprecision(1) << r.elapsed_ms << " ms\n\n";
    }

    // --- 5-asset ---
    {
        Vec S0(5, 100);
        Vec sigmas = {0.2, 0.25, 0.3, 0.22, 0.28};
        int d = 5;
        Mat corr(d, Vec(d, 0.3));
        for (int i = 0; i < d; i++) corr[i][i] = 1.0;

        std::cout << "=== 5-ASSET RAINBOW (10M paths) ===\n";
        auto r = mc_rainbow(S0, 100, 1.0, 0.05, sigmas, corr, 10'000'000, n_threads);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Price:  " << r.price << " ± " << r.stderr << "\n";
        std::cout << "Time:   " << std::setprecision(1) << r.elapsed_ms << " ms\n\n";
    }

    // --- 20-asset ---
    {
        int d = 20;
        Vec S0(d, 100);
        Vec sigmas(d);
        for (int i = 0; i < d; i++) sigmas[i] = 0.15 + 0.01 * i; // 0.15 to 0.34
        Mat corr(d, Vec(d, 0.3));
        for (int i = 0; i < d; i++) corr[i][i] = 1.0;

        std::cout << "=== 20-ASSET RAINBOW (10M paths) ===\n";
        auto r = mc_rainbow(S0, 100, 1.0, 0.05, sigmas, corr, 10'000'000, n_threads);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Price:  " << r.price << " ± " << r.stderr << "\n";
        std::cout << "Time:   " << std::setprecision(1) << r.elapsed_ms << " ms\n\n";
    }

    // --- 50-asset ---
    {
        int d = 50;
        Vec S0(d, 100);
        Vec sigmas(d);
        for (int i = 0; i < d; i++) sigmas[i] = 0.15 + 0.005 * i;
        Mat corr(d, Vec(d, 0.3));
        for (int i = 0; i < d; i++) corr[i][i] = 1.0;

        std::cout << "=== 50-ASSET RAINBOW (10M paths) ===\n";
        auto r = mc_rainbow(S0, 100, 1.0, 0.05, sigmas, corr, 10'000'000, n_threads);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Price:  " << r.price << " ± " << r.stderr << "\n";
        std::cout << "Time:   " << std::setprecision(1) << r.elapsed_ms << " ms\n\n";
    }

    // --- 100-asset ---
    {
        int d = 100;
        Vec S0(d, 100);
        Vec sigmas(d);
        for (int i = 0; i < d; i++) sigmas[i] = 0.15 + 0.002 * i; // 0.15 to 0.348
        Mat corr(d, Vec(d, 0.3));
        for (int i = 0; i < d; i++) corr[i][i] = 1.0;

        std::cout << "=== 100-ASSET RAINBOW (10M paths) ===\n";
        auto r = mc_rainbow(S0, 100, 1.0, 0.05, sigmas, corr, 10'000'000, n_threads);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Price:  " << r.price << " ± " << r.stderr << "\n";
        std::cout << "Time:   " << std::setprecision(1) << r.elapsed_ms << " ms\n";
    }

    return 0;
}
