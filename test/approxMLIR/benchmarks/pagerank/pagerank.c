// pagerank.c — pure C, pthreads, robust CLI, CSR graph
// Build: gcc -O3 -std=c11 -pthread pagerank.c -o pagerank

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

typedef struct {
    int N;                // number of nodes
    int M;                // number of edges
    int *in_row;          // CSR row pointers for in-links, size N+1
    int *in_col;          // CSR column indices (sources), size M
    int *outdeg;          // out-degree per node, size N
} GraphCSR;

typedef struct {
    int tid, P, N;
    const GraphCSR *G;
    const double *pr;     // shared current ranks (read-only during compute phase)
    double *pr_next;      // shared next ranks (write during compute phase)
    double alpha;         // damping factor (e.g., 0.85)
    double base;          // (1 - alpha)/N (recomputed if N changes, but N is fixed here)
    pthread_barrier_t *bar;
    double *dangling_sums; // length P, for per-iter reduction
    double *dp_shared;     // per-iter dangling term shared across threads
    int print;             // whether to print at the end (tid 0 will do it)
    int iters;
} WorkerArgs;

static void die(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) die("malloc");
    return p;
}

// ===== Timing util =====
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// ===== Dynamic edge buffer for file/synthetic construction =====
typedef struct {
    int *src;
    int *dst;
    int  size;
    int  cap;
} EdgeList;

static void el_init(EdgeList *el) {
    el->src = el->dst = NULL;
    el->size = 0;
    el->cap = 0;
}

static void el_push(EdgeList *el, int u, int v) {
    if (el->size == el->cap) {
        int ncap = el->cap ? el->cap * 2 : 4096;
        el->src = (int*)realloc(el->src, (size_t)ncap * sizeof(int));
        el->dst = (int*)realloc(el->dst, (size_t)ncap * sizeof(int));
        if (!el->src || !el->dst) die("realloc");
        el->cap = ncap;
    }
    el->src[el->size] = u;
    el->dst[el->size] = v;
    el->size++;
}

static void el_free(EdgeList *el) {
    free(el->src);
    free(el->dst);
    el->src = el->dst = NULL;
    el->size = el->cap = 0;
}

// ===== Build CSR (in-links) from edge list =====
static void build_csr_from_edges(const EdgeList *el, GraphCSR *G) {
    // Determine N = max node id + 1 (ensure contiguous [0..N-1])
    int max_id = -1;
    for (int i = 0; i < el->size; i++) {
        if (el->src[i] > max_id) max_id = el->src[i];
        if (el->dst[i] > max_id) max_id = el->dst[i];
    }
    G->N = (max_id >= 0) ? (max_id + 1) : 0;
    G->M = el->size;

    G->in_row = (int*)xmalloc((size_t)(G->N + 1) * sizeof(int));
    G->in_col = (int*)xmalloc((size_t)G->M * sizeof(int));
    G->outdeg = (int*)calloc((size_t)G->N, sizeof(int));
    if (!G->outdeg) die("calloc");

    // Count in-degrees and out-degrees
    int *in_deg = (int*)calloc((size_t)G->N, sizeof(int));
    if (!in_deg) die("calloc");
    for (int i = 0; i < el->size; i++) {
        int u = el->src[i], v = el->dst[i];
        if (u < 0 || v < 0) continue; // ignore invalid
        if (u >= G->N || v >= G->N) continue;
        in_deg[v]++;
        G->outdeg[u]++;
    }

    // Prefix sum to build in_row
    G->in_row[0] = 0;
    for (int v = 0; v < G->N; v++) {
        G->in_row[v+1] = G->in_row[v] + in_deg[v];
    }

    // Fill in_col with sources for each v, using a cursor array
    int *cursor = (int*)xmalloc((size_t)G->N * sizeof(int));
    memcpy(cursor, G->in_row, (size_t)G->N * sizeof(int));
    for (int i = 0; i < el->size; i++) {
        int u = el->src[i], v = el->dst[i];
        if (u < 0 || v < 0) continue;
        if (u >= G->N || v >= G->N) continue;
        int pos = cursor[v]++;
        G->in_col[pos] = u;
    }

    free(cursor);
    free(in_deg);
}

// ===== Read graph from file: lines like "u v" meaning u -> v =====
static void read_graph_file(const char *path, GraphCSR *G) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open '%s': %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }
    EdgeList el; el_init(&el);
    char *line = NULL;
    size_t len = 0;
    ssize_t nread;

    while ((nread = getline(&line, &len, fp)) != -1) {
        // skip empty/comment lines
        if (nread == 0) continue;
        // Try to parse two ints
        int u, v;
        if (sscanf(line, " %d %d", &u, &v) == 2) {
            if (u < 0 || v < 0) continue; // ignore negatives
            el_push(&el, u, v);
        } else {
            // ignore non-edge lines (headers, comments)
        }
    }
    free(line);
    fclose(fp);

    build_csr_from_edges(&el, G);
    el_free(&el);
}

// ===== Synthetic graph: create N nodes, ~DEG in-links per node (random u != v) =====
static void make_synthetic_graph(int N, int DEG, unsigned int seed, GraphCSR *G) {
    EdgeList el; el_init(&el);
    srand(seed);
    for (int v = 0; v < N; v++) {
        for (int k = 0; k < DEG; k++) {
            int u = rand() % N;
            if (u == v) u = (u + 1) % N;
            el_push(&el, u, v); // u -> v
        }
    }
    build_csr_from_edges(&el, G);
    el_free(&el);
}

// ===== Worker thread =====
static void *pagerank_worker(void *argp) {
    WorkerArgs *A = (WorkerArgs*)argp;
    int tid = A->tid, P = A->P, N = A->N;
    const GraphCSR *G = A->G;
    const double alpha = A->alpha;
    const double base = A->base;

    // Static block partition
    int start = (tid * N) / P;
    int end   = ((tid + 1) * N) / P;

    // Start barrier to align timing
    pthread_barrier_wait(A->bar);

    for (int it = 0; it < A->iters; it++) {
        // 1) Local dangling sum on this partition
        double local_dangling_sum = 0.0;
        for (int u = start; u < end; u++) {
            if (G->outdeg[u] == 0) local_dangling_sum += A->pr[u];
        }
        A->dangling_sums[tid] = local_dangling_sum;

        // 2) Reduce to shared dp = alpha * (sum dangling) / N
        pthread_barrier_wait(A->bar);
        if (tid == 0) {
            double total = 0.0;
            for (int t = 0; t < P; t++) total += A->dangling_sums[t];
            *(A->dp_shared) = alpha * (total / (double)N);
        }
        pthread_barrier_wait(A->bar);
        double dp = *(A->dp_shared);

        // 3) Compute next PR for our slice
        for (int v = start; v < end; v++) {
            double sum_in = 0.0;
            int row_start = G->in_row[v], row_end = G->in_row[v+1];
            for (int idx = row_start; idx < row_end; idx++) {
                int u = G->in_col[idx];
                int od = G->outdeg[u];
                if (od > 0) sum_in += A->pr[u] / (double)od;
            }
            A->pr_next[v] = base + dp + alpha * sum_in;
        }

        // 4) Barrier before overwriting pr
        pthread_barrier_wait(A->bar);

        // 5) Commit next -> current for our slice
        for (int v = start; v < end; v++) {
            ((double*)A->pr)[v] = A->pr_next[v];
        }

        // 6) Barrier before next iteration
        pthread_barrier_wait(A->bar);
    }

    return NULL;
}

// ===== CLI / main =====
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  -m, --mode MODE         'synthetic' (default) or 'file'\n"
        "  -f, --file PATH         edge-list file (u v per line), required for mode=file\n"
        "  -t, --threads P         number of threads (default: 1)\n"
        "  -n, --nodes N           nodes for synthetic (default: 10000)\n"
        "  -d, --degree D          ~in-degree per node for synthetic (default: 10)\n"
        "  -i, --iters K           iterations (default: 50)\n"
        "  -a, --alpha A           damping (default: 0.85)\n"
        "  -s, --seed S            RNG seed for synthetic (default: 1)\n"
        "  -p, --print             print final ranks (can be large!)\n"
        "  -h, --help              show this help\n",
        prog);
}

int main(int argc, char **argv) {
    // Defaults
    char mode[16] = "synthetic";
    char *filepath = NULL;
    int P = 1;
    int N = 10000;
    int DEG = 10;
    int iters = 50;
    double alpha = 0.85;
    unsigned int seed = 1;
    int do_print = 0;

    static struct option long_opts[] = {
        {"mode",     required_argument, 0, 'm'},
        {"file",     required_argument, 0, 'f'},
        {"threads",  required_argument, 0, 't'},
        {"nodes",    required_argument, 0, 'n'},
        {"degree",   required_argument, 0, 'd'},
        {"iters",    required_argument, 0, 'i'},
        {"alpha",    required_argument, 0, 'a'},
        {"seed",     required_argument, 0, 's'},
        {"print",    no_argument,       0, 'p'},
        {"help",     no_argument,       0, 'h'},
        {0,0,0,0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "m:f:t:n:d:i:a:s:ph", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'm': strncpy(mode, optarg, sizeof(mode)-1); mode[sizeof(mode)-1]=0; break;
            case 'f': filepath = optarg; break;
            case 't': P = atoi(optarg); break;
            case 'n': N = atoi(optarg); break;
            case 'd': DEG = atoi(optarg); break;
            case 'i': iters = atoi(optarg); break;
            case 'a': alpha = atof(optarg); break;
            case 's': seed = (unsigned int)strtoul(optarg, NULL, 10); break;
            case 'p': do_print = 1; break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }
    if (P <= 0) P = 1;
    if (alpha <= 0.0 || alpha >= 1.0) {
        fprintf(stderr, "alpha must be in (0,1), got %g\n", alpha);
        return 1;
    }

    GraphCSR G = {0};
    if (strcmp(mode, "file") == 0) {
        if (!filepath) {
            fprintf(stderr, "mode=file requires --file PATH\n");
            return 1;
        }
        read_graph_file(filepath, &G);
        if (G.N == 0) {
            fprintf(stderr, "Empty or unreadable graph file.\n");
            return 1;
        }
        fprintf(stdout, "Loaded graph from '%s': N=%d, M=%d\n", filepath, G.N, G.M);
    } else if (strcmp(mode, "synthetic") == 0) {
        if (N <= 0 || DEG < 0) {
            fprintf(stderr, "Invalid N or DEG for synthetic graph.\n");
            return 1;
        }
        make_synthetic_graph(N, DEG, seed, &G);
        fprintf(stdout, "Synthetic graph: N=%d, ~in-degree=%d, M=%d\n", G.N, DEG, G.M);
    } else {
        fprintf(stderr, "Unknown mode '%s'\n", mode);
        return 1;
    }

    // Allocate PR arrays
    double *pr = (double*)xmalloc((size_t)G.N * sizeof(double));
    double *pr_next = (double*)xmalloc((size_t)G.N * sizeof(double));

    // Initialize to uniform 1/N
    for (int v = 0; v < G.N; v++) {
        pr[v] = 1.0 / (double)G.N;
        pr_next[v] = 0.0;
    }

    // Pthreads setup
    pthread_barrier_t bar;
    if (pthread_barrier_init(&bar, NULL, (unsigned)P) != 0) die("pthread_barrier_init");

    pthread_t *threads = (pthread_t*)xmalloc((size_t)P * sizeof(pthread_t));
    WorkerArgs *args   = (WorkerArgs*)xmalloc((size_t)P * sizeof(WorkerArgs));
    double *dangling_sums = (double*)xmalloc((size_t)P * sizeof(double));
    double dp_shared = 0.0;
    double base = (1.0 - alpha) / (double)G.N;

    for (int t = 0; t < P; t++) {
        args[t].tid = t;
        args[t].P = P;
        args[t].N = G.N;
        args[t].G = &G;
        args[t].pr = pr;
        args[t].pr_next = pr_next;
        args[t].alpha = alpha;
        args[t].base = base;
        args[t].bar = &bar;
        args[t].dangling_sums = dangling_sums;
        args[t].dp_shared = &dp_shared;
        args[t].print = do_print;
        args[t].iters = iters;
        if (pthread_create(&threads[t], NULL, pagerank_worker, &args[t]) != 0) {
            die("pthread_create");
        }
    }

    double t0 = now_sec();
    // Join all
    for (int t = 0; t < P; t++) {
        pthread_join(threads[t], NULL);
    }
    double t1 = now_sec();

    printf("Time: %.6f seconds\n", t1 - t0);

    if (do_print) {
        for (int v = 0; v < G.N; v++) {
            printf("pr(%d) = %.12f\n", v, pr[v]);
        }
    }

    // Cleanup
    pthread_barrier_destroy(&bar);
    free(threads);
    free(args);
    free(dangling_sums);
    free(pr);
    free(pr_next);
    free(G.in_row);
    free(G.in_col);
    free(G.outdeg);

    return 0;
}
