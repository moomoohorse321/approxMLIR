/* Merged, OMP-independent, pure C implementation of binomial options driver.
   Builds and runs without <omp.h> or any C++ features. */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---------------- Types & constants ---------------- */

typedef double real;

#define DOUBLE 0
#define FLOAT  1
#define INT    2
#define LONG   3

/* ---------------- File I/O helpers ---------------- */

static void writeQualityFile(const char *fileName, void *ptr, int type, size_t numElements){
    FILE *fd = fopen(fileName, "wb");
    assert(fd && "Could Not Open File\n");
    fwrite(&numElements, sizeof(size_t), 1, fd);
    fwrite(&type, sizeof(int), 1, fd);
    if (type == DOUBLE) {
        fwrite(ptr, sizeof(double), numElements, fd);
    } else if (type == FLOAT) {
        fwrite(ptr, sizeof(float), numElements, fd);
    } else if (type == INT) {
        fwrite(ptr, sizeof(int), numElements, fd);
    } else {
        assert(0 && "Not supported data type to write\n");
    }
    fclose(fd);
}

static void readDataDouble(FILE *fd, double **data, size_t *numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t), 1, fd);
    size_t elements = *numElements;
    double *ptr = (double*)malloc(sizeof(double) * elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;

    int type;
    fread(&type, sizeof(int), 1, fd);
    if (type == DOUBLE) {
        fread(ptr, sizeof(double), elements, fd);
    } else if (type == FLOAT) {
        float *tmp = (float*)malloc(sizeof(float) * elements);
        fread(tmp, sizeof(float), elements, fd);
        for (size_t i = 0; i < elements; i++) ptr[i] = (double)tmp[i];
        free(tmp);
    } else if (type == INT) {
        int *tmp = (int*)malloc(sizeof(int) * elements);
        fread(tmp, sizeof(int), elements, fd);
        for (size_t i = 0; i < elements; i++) ptr[i] = (double)tmp[i];
        free(tmp);
    } else {
        assert(0 && "Unsupported input type for double array\n");
    }
}

static void readDataFloat(FILE *fd, float **data, size_t *numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t), 1, fd);
    size_t elements = *numElements;
    float *ptr = (float*)malloc(sizeof(float) * elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;

    int type;
    fread(&type, sizeof(int), 1, fd);
    if (type == FLOAT) {
        fread(ptr, sizeof(float), elements, fd);
    } else if (type == DOUBLE) {
        double *tmp = (double*)malloc(sizeof(double) * elements);
        fread(tmp, sizeof(double), elements, fd);
        for (size_t i = 0; i < elements; i++) ptr[i] = (float)tmp[i];
        free(tmp);
    } else if (type == INT) {
        int *tmp = (int*)malloc(sizeof(int) * elements);
        fread(tmp, sizeof(int), elements, fd);
        for (size_t i = 0; i < elements; i++) ptr[i] = (float)tmp[i];
        free(tmp);
    } else {
        assert(0 && "Unsupported input type for float array\n");
    }
}

static void readDataInt(FILE *fd, int **data, size_t *numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t), 1, fd);
    size_t elements = *numElements;
    int *ptr = (int*)malloc(sizeof(int) * elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;

    int type;
    fread(&type, sizeof(int), 1, fd);
    if (type == INT) {
        fread(ptr, sizeof(int), elements, fd);
    } else if (type == DOUBLE) {
        double *tmp = (double*)malloc(sizeof(double) * elements);
        fread(tmp, sizeof(double), elements, fd);
        for (size_t i = 0; i < elements; i++) ptr[i] = (int)tmp[i];
        free(tmp);
    } else if (type == FLOAT) {
        float *tmp = (float*)malloc(sizeof(float) * elements);
        fread(tmp, sizeof(float), elements, fd);
        for (size_t i = 0; i < elements; i++) ptr[i] = (int)tmp[i];
        free(tmp);
    } else {
        assert(0 && "Unsupported input type for int array\n");
    }
}

/* ---------------- Math: CND & Black-Scholes ---------------- */

static double CND(double d)
{
    const double A1 = 0.31938153;
    const double A2 = -0.356563782;
    const double A3 = 1.781477937;
    const double A4 = -1.821255978;
    const double A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double K = 1.0 / (1.0 + 0.2316419 * fabs(d));
    double cnd = RSQRT2PI * exp(-0.5 * d * d) *
                 (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));
    if (d > 0) cnd = 1.0 - cnd;
    return cnd;
}

static void BlackScholesCall(
    real *callResult,
    const real S, const real X,
    const real T, const real R,
    const real V
){
    double sqrtT = sqrt(T);
    double d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);
    double expRT = exp(-R * T);
    *callResult = (real)(S * CNDD1 - X * expRT * CNDD2);
}

/* ---------------- Binomial tree (sequential) ---------------- */

static inline double expiryCallValue(const int num_steps, double S, double X, double vDt, int i)
{
    double d = S * exp(vDt * (2.0 * i - num_steps)) - X;
    return (d > 0.0) ? d : 0.0;
}

static real binomialOptionValue(
    const int num_steps,
    const real S, const real X,
    const real T, const real R,
    const real V
){
    const double dt = T / (double)num_steps;
    const double vDt = V * sqrt(dt);
    const double rDt = R * dt;

    const double If = exp(rDt);
    const double Df = exp(-rDt);

    const double u = exp(vDt);
    const double d = exp(-vDt);
    const double pu = (If - d) / (u - d);
    const double pd = 1.0 - pu;
    const double puByDf = pu * Df;
    const double pdByDf = pd * Df;

    /* values at expiration */
    double *call = (double*)malloc((size_t)(num_steps + 1) * sizeof(double));
    assert(call && "malloc failed");

    for (int i = 0; i <= num_steps; ++i) {
        call[i] = expiryCallValue(num_steps, S, X, vDt, i);
    }

    /* backward induction */
    for (int i = num_steps; i > 0; --i) {
        for (int j = 0; j <= i - 1; ++j) {
            call[j] = puByDf * call[j + 1] + pdByDf * call[j];
        }
    }

    real result = (real)call[0];
    free(call);
    return result;
}

/* ---------------- Main ---------------- */

int main(int argc, char **argv)
{
    if (!(argc == 4 || argc == 5)) {
        printf("USAGE: %s input_file num_steps seed [output_file]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *inputFile = argv[1];
    const int num_steps = atoi(argv[2]);
    const int seed = atoi(argv[3]);
    (void)seed; /* Not used, but keep for interface parity; call srand if needed */
    srand((unsigned)seed);

    FILE *file = fopen(inputFile, "rb");
    if (file == NULL) {
        printf("ERROR: Unable to open file `%s`.\n", inputFile);
        return EXIT_FAILURE;
    }

    int write_output = 0;
    const char *ofname = NULL;
    if (argc == 5) {
        write_output = 1;
        ofname = argv[4];
    }

    /* Read inputs: otype (unused), then S, X, R, V, T */
    size_t numOptions = 0;

    int *otype = NULL;
    readDataInt(file, &otype, &numOptions);

    real *S = NULL, *X = NULL, *R = NULL, *V = NULL, *T = NULL;
    {
        double *Sd = NULL, *Xd = NULL, *Rd = NULL, *Vd = NULL, *Td = NULL;
        size_t n2=0,n3=0,n4=0,n5=0,n6=0;

        readDataDouble(file, &Sd, &n2);
        readDataDouble(file, &Xd, &n3);
        readDataDouble(file, &Rd, &n4);
        readDataDouble(file, &Vd, &n5);
        readDataDouble(file, &Td, &n6);

        /* Basic sanity: sizes should match */
        assert(n2 == numOptions && n3 == numOptions && n4 == numOptions &&
               n5 == numOptions && n6 == numOptions);

        S = (real*)malloc(numOptions * sizeof(real));
        X = (real*)malloc(numOptions * sizeof(real));
        R = (real*)malloc(numOptions * sizeof(real));
        V = (real*)malloc(numOptions * sizeof(real));
        T = (real*)malloc(numOptions * sizeof(real));
        assert(S && X && R && V && T);

        for (size_t i = 0; i < numOptions; ++i) {
            S[i] = (real)Sd[i];
            X[i] = (real)Xd[i];
            R[i] = (real)Rd[i];
            V[i] = (real)Vd[i];
            T[i] = (real)Td[i];
        }

        free(Sd); free(Xd); free(Rd); free(Vd); free(Td);
    }

    fclose(file);

    /* Original code overwrote V with 0.10f — keep behavior for functional parity */
    for (size_t i = 0; i < numOptions; ++i) {
        V[i] = 0.10;
    }

    free(otype); /* Unused thereafter */

    const int NUM_OPTIONS = (int)numOptions;
    real *callValueBS = (real*)malloc((size_t)NUM_OPTIONS * sizeof(real));
    real *callValue   = (real*)malloc((size_t)NUM_OPTIONS * sizeof(real));
    assert(callValueBS && callValue);

    printf("[%s] - Starting with %d options and %d iterations...\n",
           argv[0], NUM_OPTIONS, num_steps);

    /* Black-Scholes baseline */
    for (int i = 0; i < NUM_OPTIONS; ++i) {
        BlackScholesCall(&callValueBS[i], S[i], X[i], T[i], R[i], V[i]);
    }

    /* Sequential binomial */
    clock_t t0 = clock();
    for (int i = 0; i < NUM_OPTIONS; ++i) {
        callValue[i] = binomialOptionValue(num_steps, S[i], X[i], T[i], R[i], V[i]);
    }
    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("binomialOptions(sequential) time: %f sec\n", secs);
    if (secs > 0.0) {
        printf("Options per second            : %f\n", (double)NUM_OPTIONS / secs);
    }

    /* Compare */
    printf("Comparing the results...\n");
    printf("CPU binomial vs. Black-Scholes\n");
    double sumDelta = 0.0, sumRef = 0.0;
    for (int i = 0; i < NUM_OPTIONS; ++i) {
        sumDelta += fabs((double)callValueBS[i] - (double)callValue[i]);
        sumRef   += fabs((double)callValueBS[i]);
    }
    if (sumRef > 1E-5) {
        printf("L1 norm: %E\n", sumDelta / sumRef);
    } else {
        printf("Avg. diff: %E\n", (double)(sumDelta / (double)NUM_OPTIONS));
    }

    if (write_output && ofname) {
        writeQualityFile(ofname, callValue, DOUBLE, (size_t)NUM_OPTIONS);
    }

    free(S); free(X); free(R); free(V); free(T);
    free(callValueBS);
    free(callValue);

    return EXIT_SUCCESS;
}
