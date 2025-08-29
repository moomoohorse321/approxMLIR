// make_data.cpp — produces input compatible with merged_binomial.c
// Format per block: [size_t numElements][int type_code][array bytes]
// Blocks (in order): otype(INT), S(DOUBLE), X(DOUBLE), R(DOUBLE), V(DOUBLE), T(DOUBLE)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <vector>
#include <random>
#include <iostream>

enum { DOUBLE = 0, FLOAT = 1, INT = 2, LONG_ = 3 };

static void write_block(FILE* f, const void* data, size_t count, int type_code, size_t elem_size) {
    if (!f) { std::fprintf(stderr, "Invalid file handle\n"); std::exit(1); }
    if (std::fwrite(&count, sizeof(size_t), 1, f) != 1) { std::perror("fwrite"); std::exit(1); }
    if (std::fwrite(&type_code, sizeof(int), 1, f) != 1) { std::perror("fwrite"); std::exit(1); }
    if (count && std::fwrite(data, elem_size, count, f) != count) { std::perror("fwrite"); std::exit(1); }
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::fprintf(stderr, "USAGE: %s <output_file> <num_options> [seed]\n", argv[0]);
        return 2;
    }
    const char* out_path = argv[1];
    const size_t N = static_cast<size_t>(std::strtoull(argv[2], nullptr, 10));
    const uint64_t seed = (argc >= 4) ? std::strtoull(argv[3], nullptr, 10) : 42ull;

    if (N == 0) {
        std::fprintf(stderr, "num_options must be > 0\n");
        return 2;
    }

    std::mt19937_64 rng(seed);

    // Distributions (plausible ranges, avoid pathological values)
    std::uniform_real_distribution<double> dist_S(5.0, 200.0);      // spot
    std::uniform_real_distribution<double> dist_Kratio(0.5, 1.5);   // strike as ratio of S
    std::uniform_real_distribution<double> dist_R(0.00, 0.10);      // 0%..10%
    std::uniform_real_distribution<double> dist_V(0.05, 0.60);      // 5%..60% (will be overwritten to 0.10 by solver)
    std::uniform_real_distribution<double> dist_T(0.25, 2.00);      // 3 months..2 years

    std::vector<int>    otype(N, 0);        // 0 = call (unused by solver, but included for completeness)
    std::vector<double> S(N), X(N), R(N), V(N), T(N);

    for (size_t i = 0; i < N; ++i) {
        const double s = dist_S(rng);
        const double x = s * dist_Kratio(rng);
        S[i] = s;
        X[i] = (x < 1e-6 ? 1e-6 : x);
        R[i] = dist_R(rng);
        V[i] = dist_V(rng);
        T[i] = dist_T(rng);
    }

    FILE* f = std::fopen(out_path, "wb");
    if (!f) {
        std::perror("fopen");
        return 1;
    }

    // Write blocks in the exact order the solver reads them.
    write_block(f, otype.data(), N, INT,    sizeof(int));
    write_block(f, S.data(),     N, DOUBLE, sizeof(double));
    write_block(f, X.data(),     N, DOUBLE, sizeof(double));
    write_block(f, R.data(),     N, DOUBLE, sizeof(double));
    write_block(f, V.data(),     N, DOUBLE, sizeof(double));
    write_block(f, T.data(),     N, DOUBLE, sizeof(double));

    if (std::fclose(f) != 0) {
        std::perror("fclose");
        return 1;
    }

    std::cout << "Wrote " << N << " options to " << out_path << " (seed " << seed << ")\n";
    std::cout << "Blocks: otype(INT), S/X/R/V/T (DOUBLE). "
                 "Note: your solver sets V=0.10 internally for parity.\n";
    return 0;
}
