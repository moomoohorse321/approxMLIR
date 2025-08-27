// RUN: cgeist -O0 %stdinclude %s -S > %s.mlir
// RUN: cgeist -O0 %stdinclude %s -o %s.exec -lm

// // Knob — CNDF substitution (proper func_substitute)
// "approxMLIR.util.annotation.decision_tree"() <{
//   func_name = "compute_cndf",
//   transform_type = "func_substitute",
//   num_thresholds = 1 : i32,
//   thresholds_uppers = array<i32: 10>,
//   thresholds_lowers = array<i32: 0>,
//   decision_values = array<i32: 0, 1>,
//   thresholds = array<i32: 5>,
//   decisions = array<i32: 0, 1>
// }> : () -> ()


// // Required for func_substitute
// "approxMLIR.util.annotation.convert_to_call"() <{func_name = "compute_cndf"}> : () -> ()

// // Knob — Black-Scholes approximation (func_substitute)
// "approxMLIR.util.annotation.decision_tree"() <{
//   func_name = "BlkSchlsEqEuroNoDiv",
//   transform_type = "func_substitute",
//   num_thresholds = 1 : i32,
//   thresholds_uppers = array<i32: 10>,
//   thresholds_lowers = array<i32: 0>,
//   decision_values = array<i32: 0, 1>,
//   thresholds = array<i32: 5>,
//   decisions = array<i32: 0, 1>
// }> : () -> ()

// // Required for func_substitute
// "approxMLIR.util.annotation.convert_to_call"() <{func_name = "BlkSchlsEqEuroNoDiv"}> : () -> ()


// ------------------------------- C code -------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define TYPE_DOUBLE 0
#define TYPE_FLOAT  1
#define TYPE_INT    2

typedef double fptype;

// -------------------- Globals --------------------
fptype *prices = NULL;
size_t  numOptions = 0;

int    *otype = NULL;
fptype *sptprice = NULL;
fptype *strike = NULL;
fptype *rate = NULL;
fptype *volatility = NULL;
fptype *otime = NULL;

// -------------------- Small utility --------------------
static void die(const char* msg) {
    fprintf(stderr, "%s\n", msg);
    exit(1);
}

// Computes caller-side states summarizing approximation safety:
static inline int decide_cndf_state(fptype x) {
    // Near 0 → safe to approximate, tails → exact
    return (fabs(x) > 2.0 ? 2 : fabs(x) > 1.0 ? 1 : 0);
}

static inline int decide_bs_state(fptype s, fptype K, fptype r, fptype t) {
    // ATM vs OTM and discount factor sensitivity
    fptype a = fabs(s / K - 1.0);   // distance from ATM
    fptype b = fabs(r) * t;         // discount impact
    return (a > 0.30) + (b > 0.20); // 0 = safe, 1 = partial, 2 = exact
}


// -------------------- File I/O helpers --------------------
void writeQualityFile(char *fileName, void *ptr, int type, size_t numElements){
    FILE *fd = fopen(fileName, "wb");
    if (!fd) die("Could Not Open File for writing");
    if (fwrite(&numElements, sizeof(size_t), 1, fd) != 1) die("write error");
    if (fwrite(&type, sizeof(int), 1, fd) != 1) die("write error");
    if (type == TYPE_DOUBLE) {
        if (fwrite(ptr, sizeof(double), numElements, fd) != numElements) die("write error");
    } else if (type == TYPE_FLOAT) {
        if (fwrite(ptr, sizeof(float), numElements, fd) != numElements) die("write error");
    } else if (type == TYPE_INT) {
        if (fwrite(ptr, sizeof(int), numElements, fd) != numElements) die("write error");
    } else {
        die("Unsupported data type to write");
    }
    fclose(fd);
}

// Read into double output (fptype) regardless of on-disk type
void readDataAsDouble(FILE *fd, fptype **data, size_t *numElements){
    if (!fd) die("File pointer is not valid");
    if (fread(numElements, sizeof(size_t), 1, fd) != 1) die("read error");
    size_t n = *numElements;
    fptype *ptr = (fptype*) malloc(sizeof(fptype)*n);
    if (!ptr) die("Could not allocate");
    *data = ptr;
    int type;
    if (fread(&type, sizeof(int), 1, fd) != 1) die("read error");
    if (type == TYPE_DOUBLE){
        if (fread(ptr, sizeof(double), n, fd) != n) die("read error");
    } else if (type == TYPE_FLOAT){
        float *tmp = (float*) malloc(sizeof(float)*n);
        if (!tmp) die("alloc fail");
        if (fread(tmp, sizeof(float), n, fd) != n) die("read error");
        for (size_t i=0;i<n;++i) ptr[i] = (fptype)tmp[i];
        free(tmp);
    } else if (type == TYPE_INT){
        int *tmp = (int*) malloc(sizeof(int)*n);
        if (!tmp) die("alloc fail");
        if (fread(tmp, sizeof(int), n, fd) != n) die("read error");
        for (size_t i=0;i<n;++i) ptr[i] = (fptype)tmp[i];
        free(tmp);
    } else {
        die("Unsupported data type in file");
    }
}

// Read into int output regardless of on-disk type
void readDataAsInt(FILE *fd, int **data, size_t *numElements){
    if (!fd) die("File pointer is not valid");
    if (fread(numElements, sizeof(size_t), 1, fd) != 1) die("read error");
    size_t n = *numElements;
    int *ptr = (int*) malloc(sizeof(int)*n);
    if (!ptr) die("Could not allocate");
    *data = ptr;
    int type;
    if (fread(&type, sizeof(int), 1, fd) != 1) die("read error");
    if (type == TYPE_INT){
        if (fread(ptr, sizeof(int), n, fd) != n) die("read error");
    } else if (type == TYPE_DOUBLE){
        double *tmp = (double*) malloc(sizeof(double)*n);
        if (!tmp) die("alloc fail");
        if (fread(tmp, sizeof(double), n, fd) != n) die("read error");
        for (size_t i=0;i<n;++i) ptr[i] = (int)tmp[i];
        free(tmp);
    } else if (type == TYPE_FLOAT){
        float *tmp = (float*) malloc(sizeof(float)*n);
        if (!tmp) die("alloc fail");
        if (fread(tmp, sizeof(float), n, fd) != n) die("read error");
        for (size_t i=0;i<n;++i) ptr[i] = (int)tmp[i];
        free(tmp);
    } else {
        die("Unsupported data type in file");
    }
}

// -------------------- Math constants --------------------
static const fptype inv_sqrt_2xPI = 0.39894228040143270286;
static const fptype zero  = 0.0;
static const fptype half  = 0.5;
static const fptype const1= 0.2316419;
static const fptype one   = 1.0;
static const fptype const2= 0.319381530;
static const fptype const3= 0.356563782;
static const fptype const4= 1.781477937;
static const fptype const5= 1.821255978;
static const fptype const6= 1.330274429;

// exact CNDF
fptype compute_cndf(fptype x, int state) {
    int sign = 0;
    if (x < zero) { x = -x; sign = 1; }
    fptype nprime = exp(-half * x * x) * inv_sqrt_2xPI;
    fptype k = one / (one + const1 * x);
    fptype k2 = k*k, k3 = k2*k, k4 = k3*k, k5 = k4*k;
    fptype poly = k*const2 + k2*(-const3) + k3*const4 + k4*(-const5) + k5*const6;
    fptype cdf = one - poly*nprime;
    return sign ? (one - cdf) : cdf;
}

// approximate CNDF (fewer terms)
fptype approx_compute_cndf(fptype x, int state) {
    int sign = 0;
    if (x < zero) { x = -x; sign = 1; }
    fptype nprime = exp(-half * x * x) * inv_sqrt_2xPI;
    fptype k = one / (one + const1 * x);
    fptype k2 = k*k, k3 = k2*k;
    fptype poly = k*const2 + k2*(-const3) + k3*const4; // truncated series
    fptype cdf = one - poly*nprime;
    return sign ? (one - cdf) : cdf;
}

// Approximate Black-Scholes pricing: faster but less accurate
fptype approx_BlkSchlsEqEuroNoDiv(fptype s, fptype K, fptype r,
                                  fptype v, fptype t, int state) {
    // Approx sqrt is okay
    fptype sqrtT = sqrt(t);

    // Approximate log with a fast series: log(1+x) ≈ x - x²/2 for |x| small
    // Fall back to normal log if ratio far from 1 to avoid huge errors
    fptype ratio = s / K;
    fptype logTerm;
    if (fabs(ratio - 1.0) < 0.3) {
        fptype x = ratio - 1.0;
        logTerm = x - (x * x) * 0.5;
    } else {
        logTerm = log(ratio);
    }

    // Same d1/d2 formulas
    fptype d1 = ((r + (v*v)*half) * t + logTerm) / (v * sqrtT);
    fptype d2 = d1 - v * sqrtT;

    // Approximate exp(-r*t) with 1 - r*t for small t
    fptype discK;
    if (r * t < 0.2)
        discK = (1.0 - r * t) * K;
    else
        discK = exp(-r * t) * K;

    int cndf_state_d1 = decide_cndf_state(d1);
    int cndf_state_d2 = decide_cndf_state(d2);

    // Use exact CNDF for now, though you could chain its approximation knob too
    fptype Nd1 = compute_cndf(d1, cndf_state_d1);
    fptype Nd2 = compute_cndf(d2, cndf_state_d1);

    return discK * (1.0 - Nd2) - s * (1.0 - Nd1);
}


// -------------------- BS pricing core --------------------
static inline fptype BlkSchlsEqEuroNoDiv(fptype s, fptype K, fptype r,
                                         fptype v, fptype t, int state){
    fptype sqrtT  = sqrt(t);
    fptype logTerm= log(s / K);
    fptype d1 = ((r + (v*v)*half) * t + logTerm) / (v * sqrtT);
    fptype d2 = d1 - v * sqrtT;
    fptype Nd1 = compute_cndf(d1, 0);   // knob can substitute this call
    fptype Nd2 = compute_cndf(d2, 0);
    fptype discK = exp(-r * t) * K;
    // Put price (matches your original path)
    return discK * (1.0 - Nd2) - s * (1.0 - Nd1);
}

static inline void price_range(size_t begin, size_t end){
    size_t i;
    for (i = begin; i < end; ++i){
        int bs_state = decide_bs_state(sptprice[i], strike[i], rate[i], otime[i]);
        prices[i] = BlkSchlsEqEuroNoDiv(sptprice[i], strike[i], rate[i], volatility[i], otime[i], bs_state);
    }
}

// -------------------- Text results helpers --------------------
static inline void dump_text_rows(FILE* tf,
                                  const int* _otype,
                                  const fptype* _sptprice,
                                  const fptype* _strike,
                                  const fptype* _rate,
                                  const fptype* _volatility,
                                  const fptype* _otime,
                                  const fptype* _prices,
                                  size_t N)
{
    size_t i;
    for (i = 0; i < N; ++i) {
        fprintf(tf, "%zu %d %.17g %.17g %.17g %.17g %.17g %.17g\n",
                i, _otype[i], _sptprice[i], _strike[i], _rate[i],
                _volatility[i], _otime[i], _prices[i]);
    }
}

// Human-readable text dump (kept exact)
static void writeTextResults(const char* txtPath,
                             const int* _otype,
                             const fptype* _sptprice,
                             const fptype* _strike,
                             const fptype* _rate,
                             const fptype* _volatility,
                             const fptype* _otime,
                             const fptype* _prices,
                             size_t N)
{
    FILE* tf = fopen(txtPath, "w");
    if (!tf) die("fopen text dump failed");
    fprintf(tf, "# idx otype sptprice strike rate volatility otime price\n");
    dump_text_rows(tf, _otype, _sptprice, _strike, _rate, _volatility, _otime, _prices, N);
    fclose(tf);
}


// -------------------- Compute (serial) --------------------
int bs_thread(){
    clock_t t0 = clock();
    price_range(0, numOptions);  
    clock_t t1 = clock();
    double secs = (double)(t1 - t0) / CLOCKS_PER_SEC;
    printf("Elapsed: %f\n", secs);
    return 0;
}

// -------------------- Main --------------------
/**
 * run this test by:
 *  ../llvm-project/build/bin/llvm-lit ~/Polygeist/build/tools/cgeist/Test/approxMLIR/approx_blackscholes.c
 * (adapt the Polygeist path to your local build)
 */
int main(int argc, char **argv){
    printf("PARSEC Benchmark Suite\n");
    if (argc != 4){
        printf("Usage:\n\t%s <nthreads> <inputFile> <outputFile>\n", argv[0]);
        return 1;
    }
    {
        int nThreads = atoi(argv[1]);
        if (nThreads != 1){
            printf("Error: <nthreads> must be 1 (serial version)\n");
            return 1;
        }
    }

    char *inputFile  = argv[2];
    char *outputFile = argv[3];

    FILE *file = fopen(inputFile, "rb");
    if (!file){
        printf("ERROR: Unable to open file `%s`.\n", inputFile);
        return 1;
    }

    // Read input arrays
    readDataAsInt(file, &otype,      &numOptions);
    readDataAsDouble(file, &sptprice,&numOptions);
    readDataAsDouble(file, &strike,  &numOptions);
    readDataAsDouble(file, &rate,    &numOptions);
    readDataAsDouble(file, &volatility,&numOptions);
    readDataAsDouble(file, &otime,   &numOptions);
    fclose(file);

    prices = (fptype*) malloc(sizeof(fptype)*numOptions);
    if (!prices) die("alloc prices failed");
    memset(prices, 0, sizeof(fptype)*numOptions);

    bs_thread();

    // Binary output (unchanged)
    writeQualityFile(outputFile, prices, TYPE_DOUBLE, numOptions);

    // Text output (via task-skipping wrapper)
    {
        size_t len = strlen(outputFile);
        char *txtPath = (char*)malloc(len + 5); // ".txt" + NUL
        if (!txtPath) die("alloc txtPath failed");
        memcpy(txtPath, outputFile, len);
        memcpy(txtPath + len, ".txt", 5);
        writeTextResults(txtPath,
                         otype, sptprice, strike, rate, volatility, otime,
                         prices, numOptions);
        free(txtPath);
    }

    // Cleanup
    free(sptprice);
    free(strike);
    free(rate);
    free(volatility);
    free(otime);
    free(otype);
    free(prices);
    return 0;
}
