// gen_bs_input.cpp
// Generate input for the Black-Scholes program (no OpenMP).
// Layout matches readData() usage: six arrays written sequentially:
// 1) otype (int), 2) sptprice (double), 3) strike (double),
// 4) rate (double), 5) volatility (double), 6) otime (double).
// Each array is: [size_t numElements][int typeCode][raw array bytes].

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <iostream>

#define TYPE_DOUBLE 0
#define TYPE_FLOAT  1
#define TYPE_INT    2

static void writeArrayHeader(FILE* f, size_t n, int typeCode) {
    if (std::fwrite(&n, sizeof(size_t), 1, f) != 1) {
        std::perror("fwrite numElements"); std::exit(1);
    }
    if (std::fwrite(&typeCode, sizeof(int), 1, f) != 1) {
        std::perror("fwrite typeCode"); std::exit(1);
    }
}

static void writeIntArray(FILE* f, const std::vector<int>& v) {
    writeArrayHeader(f, v.size(), TYPE_INT);
    if (!v.empty() && std::fwrite(v.data(), sizeof(int), v.size(), f) != v.size()) {
        std::perror("fwrite int array"); std::exit(1);
    }
}

static void writeDoubleArray(FILE* f, const std::vector<double>& v) {
    writeArrayHeader(f, v.size(), TYPE_DOUBLE);
    if (!v.empty() && std::fwrite(v.data(), sizeof(double), v.size(), f) != v.size()) {
        std::perror("fwrite double array"); std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <outputFile> <numOptions> [seed]\n";
        std::cerr << "Example: " << argv[0] << " input.bin 100000 42\n";
        return 1;
    }

    const std::string outPath = argv[1];
    const size_t N = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
    uint64_t seed = (argc == 4) ? static_cast<uint64_t>(std::strtoull(argv[3], nullptr, 10))
                                : std::random_device{}();

    FILE* f = std::fopen(outPath.c_str(), "wb");
    if (!f) { std::perror(("fopen " + outPath).c_str()); return 1; }

    // RNG setup
    std::mt19937_64 rng(seed);

    // Distributions (adjust ranges as needed)
    // Spot price around 100 ± 30
    std::normal_distribution<double> spotDist(100.0, 30.0);
    // Strike around 100 ± 25
    std::normal_distribution<double> strikeDist(100.0, 25.0);
    // Risk-free rate between 0% and 10%
    std::uniform_real_distribution<double> rateDist(0.00, 0.10);
    // Volatility between 5% and 80%
    std::uniform_real_distribution<double> volDist(0.05, 0.80);
    // Time to maturity between 1/365 and 3 years
    std::uniform_real_distribution<double> timeDist(1.0/365.0, 3.0);
    // Option type 0/1 (e.g., call/put—solver doesn’t use it, but we include it)
    std::bernoulli_distribution otypeDist(0.5);

    std::vector<int>    otype(N);
    std::vector<double> sptprice(N), strike(N), rate(N), volatility(N), otime(N);

    auto clamp_pos = [](double x, double minv) {
        return (x < minv) ? minv : x;
    };

    for (size_t i = 0; i < N; ++i) {
        // Ensure strictly positive prices/strikes (avoid pathological values)
        double S = clamp_pos(spotDist(rng), 1e-6);
        double K = clamp_pos(strikeDist(rng), 1e-6);

        // Occasionally tie strike closer to spot to make realistic instances
        if ((i % 7) == 0) {
            K = clamp_pos(S * std::uniform_real_distribution<double>(0.8, 1.2)(rng), 1e-6);
        }

        otype[i]     = otypeDist(rng) ? 1 : 0;
        sptprice[i]  = S;
        strike[i]    = K;
        rate[i]      = rateDist(rng);
        volatility[i]= volDist(rng);
        otime[i]     = timeDist(rng);
    }

    // Write arrays in the exact order your solver reads them:
    // otype, sptprice, strike, rate, volatility, otime
    writeIntArray(f, otype);
    writeDoubleArray(f, sptprice);
    writeDoubleArray(f, strike);
    writeDoubleArray(f, rate);
    writeDoubleArray(f, volatility);
    writeDoubleArray(f, otime);

    std::fclose(f);

    std::cout << "Wrote " << N << " options to " << outPath
              << " (seed=" << seed << ")\n";
    std::cout << "Layout: [size_t, int] header + raw array, repeated 6 times.\n";
    return 0;
}
