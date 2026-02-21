#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <getopt.h>

int approx_state_identity(int state) { return state; }

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
    int state;
} WorkerArgs;

typedef struct {
    int id;
    double rank;
} PageRankEntry;

static int compare_pagerank(const void *a, const void *b) {
    double rank_a = ((const PageRankEntry*)a)->rank;
    double rank_b = ((const PageRankEntry*)b)->rank;
    if (rank_a < rank_b) return 1;
    if (rank_a > rank_b) return -1;
    return 0;
}

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

    int *in_deg = (int*)calloc((size_t)G->N, sizeof(int));
    if (!in_deg) die("calloc");
    for (int i = 0; i < el->size; i++) {
        int u = el->src[i], v = el->dst[i];
        if (u < 0 || v < 0) continue;
        if (u >= G->N || v >= G->N) continue;
        in_deg[v]++;
        G->outdeg[u]++;
    }

    G->in_row[0] = 0;
    for (int v = 0; v < G->N; v++) {
        G->in_row[v+1] = G->in_row[v] + in_deg[v];
    }

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
        if (nread == 0) continue;
        int u, v;
        if (sscanf(line, " %d %d", &u, &v) == 2) {
            if (u < 0 || v < 0) continue;
            el_push(&el, u, v);
        } else {
            // ignore non-edge lines
        }
    }
    free(line);
    fclose(fp);

    build_csr_from_edges(&el, G);
    el_free(&el);
}

// ===== Synthetic graph =====
static void make_synthetic_graph(int N, int DEG, unsigned int seed, GraphCSR *G) {
    EdgeList el; el_init(&el);
    srand(seed);
    for (int v = 0; v < N; v++) {
        for (int k = 0; k < DEG; k++) {
            int u = rand() % N;
            if (u == v) u = (u + 1) % N;
            el_push(&el, u, v);
        }
    }
    build_csr_from_edges(&el, G);
    el_free(&el);
}

/* =====================  APPROX-READY KERNELS  ===================== */

/* Knob A target — inner accumulation over in-neighbors (loop_perforate).
   NOTE: state is forwarded only; not used in logic (pass responsibility to approxMLIR). */
double compute_sum_in_neighbors(const GraphCSR *G,
                                              const double *pr,
                                              int v,
                                              int state) {
    double sum_in = 0.0;
    int row_start = G->in_row[v];
    int row_end   = G->in_row[v+1];
    for (int idx = row_start; idx < row_end; idx++) {
        int u = G->in_col[idx];
        int od = G->outdeg[u];
        if (od > 0) sum_in += pr[u] / (double)od;
    }
    return sum_in;
}
double approx_update_node_rank_1(const GraphCSR *G,
                                             const double *pr,
                                             int v,
                                             double alpha,
                                             double base,
                                             double dp, int indeg,
                                             int state) {
    /* Simple fast approximation: sample every other in-neighbor (stride 2).
       NOTE: do not branch on `state` here; approxMLIR controls substitution. */
    double sum_in = 0.0;
    int row_start = G->in_row[v];
    int row_end   = G->in_row[v+1];
    int span = row_end - row_start;

    for (int idx = row_start; idx < row_end; idx +=2) {
        int u = G->in_col[idx];
        int od = G->outdeg[u];
        if (od > 0) sum_in += pr[u] / (double)od;
    }
    // Re-scale the sampled sum back (very rough, but preserves expectation)
    sum_in = sum_in * 2;
    return base + dp + alpha * sum_in;
}

/* Approximate alternative for Knob B (func_substitute).
   Keep contract identical; still forwards state to nested knob if used. */

double approx_update_node_rank_2(const GraphCSR *G,
                                             const double *pr,
                                             int v,
                                             double alpha,
                                             double base,
                                             double dp, int indeg,
                                             int state) {
    /* Simple fast approximation: sample every other in-neighbor (stride 2).
       NOTE: do not branch on `state` here; approxMLIR controls substitution. */
    double sum_in = 0.0;
    int row_start = G->in_row[v];
    int row_end   = G->in_row[v+1];
    int span = row_end - row_start;

    for (int idx = row_start; idx < row_end; idx +=3) {
        int u = G->in_col[idx];
        int od = G->outdeg[u];
        if (od > 0) sum_in += pr[u] / (double)od;
    }
    // Re-scale the sampled sum back (very rough, but preserves expectation)
    sum_in = sum_in * 3;
    return base + dp + alpha * sum_in;
}

/* Knob B target — per-node PageRank update (func_substitute).
   exact version; last arg is the knobbing state, forwarded to nested knob */
// @approx:decision_tree {
//   transform_type: func_substitute
//   state_indices: [7]
//   state_function: approx_state_identity
//   thresholds: [24]
//   thresholds_lower: [1]
//   thresholds_upper: [50]
//   decisions: [0, 0]
//   decision_values: [0, 1, 2]
// }
double update_node_rank(const GraphCSR *G,
                                      const double *pr,
                                      int v,
                                      double alpha,
                                      double base,
                                      double dp, int indeg,
                                      int state) {
    double sum_in = compute_sum_in_neighbors(G, pr, v, indeg);
    return base + dp + alpha * sum_in;
}

/* =====================  WORKER / PARALLEL CODE  ===================== */

void *approx_pagerank_worker_impl_1(void *argp, int state) {
    WorkerArgs *A = (WorkerArgs*)argp;
    int tid = A->tid, P = A->P, N = A->N;
    const GraphCSR *G = A->G;
    const double alpha = A->alpha;
    const double base = A->base;

    int start = (tid * N) / P;
    int end   = ((tid + 1) * N) / P;

    for (int it = 0; it < A->iters; it ++) {
        if(rand() % 100 < 25) continue;
        // 1) Local dangling sum
        double local_dangling_sum = 0.0;
        for (int u = start; u < end; u++) {
            if (G->outdeg[u] == 0) local_dangling_sum += A->pr[u];
        }
        A->dangling_sums[tid] = local_dangling_sum;

        // 2) Reduce to shared dp = alpha * (sum dangling) / N
        if (tid == 0) {
            double total = 0.0;
            for (int t = 0; t < P; t++) total += A->dangling_sums[t];
            *(A->dp_shared) = alpha * (total / (double)N);
        }
        double dp = *(A->dp_shared);

        // 3) Compute next PR for our slice using the knobbed update function
        for (int v = start; v < end; v++) {
            int indeg = G->in_row[v+1] - G->in_row[v];
            // Exact path (approxMLIR may substitute this with approx_update_node_rank)
            A->pr_next[v] = update_node_rank(G, A->pr, v, alpha, base, dp, indeg, it);
        }

        // 5) Commit next -> current for our slice
        for (int v = start; v < end; v++) {
            ((double*)A->pr)[v] = A->pr_next[v];
        }
    }

    return NULL;
}

void *approx_pagerank_worker_impl_2(void *argp, int state) {
    WorkerArgs *A = (WorkerArgs*)argp;
    int tid = A->tid, P = A->P, N = A->N;
    const GraphCSR *G = A->G;
    const double alpha = A->alpha;
    const double base = A->base;

    int start = (tid * N) / P;
    int end   = ((tid + 1) * N) / P;

    for (int it = 0; it < A->iters; it += 2) {
        // 1) Local dangling sum
        double local_dangling_sum = 0.0;
        for (int u = start; u < end; u++) {
            if (G->outdeg[u] == 0) local_dangling_sum += A->pr[u];
        }
        A->dangling_sums[tid] = local_dangling_sum;

        // 2) Reduce to shared dp = alpha * (sum dangling) / N
        if (tid == 0) {
            double total = 0.0;
            for (int t = 0; t < P; t++) total += A->dangling_sums[t];
            *(A->dp_shared) = alpha * (total / (double)N);
        }
        double dp = *(A->dp_shared);

        // 3) Compute next PR for our slice using the knobbed update function
        for (int v = start; v < end; v++) {
            int indeg = G->in_row[v+1] - G->in_row[v];
            // Exact path (approxMLIR may substitute this with approx_update_node_rank)
            A->pr_next[v] = update_node_rank(G, A->pr, v, alpha, base, dp, indeg, it);
        }

        // 5) Commit next -> current for our slice
        for (int v = start; v < end; v++) {
            ((double*)A->pr)[v] = A->pr_next[v];
        }
    }

    return NULL;
}

// @approx:decision_tree {
//   transform_type: func_substitute
//   state_indices: [1]
//   state_function: approx_state_identity
//   thresholds: [3]
//   thresholds_lower: [1]
//   thresholds_upper: [5]
//   decisions: [0, 0]
//   decision_values: [0, 1, 2]
// }
void *pagerank_worker_impl(void *argp, int state) {
    WorkerArgs *A = (WorkerArgs*)argp;
    int tid = A->tid, P = A->P, N = A->N;
    const GraphCSR *G = A->G;
    const double alpha = A->alpha;
    const double base = A->base;

    int start = (tid * N) / P;
    int end   = ((tid + 1) * N) / P;

    pthread_barrier_wait(A->bar);

    for (int it = 0; it < A->iters; it++) {
        // 1) Local dangling sum
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

        // 3) Compute next PR for our slice using the knobbed update function
        for (int v = start; v < end; v++) {
            int indeg = G->in_row[v+1] - G->in_row[v];
            // Exact path (approxMLIR may substitute this with approx_update_node_rank)
            A->pr_next[v] = update_node_rank(G, A->pr, v, alpha, base, dp, indeg, it);
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

void *pagerank_worker(void *argp) {
    WorkerArgs *A = (WorkerArgs*)argp;
    return pagerank_worker_impl(argp, A->state);
}

/* =====================  CLI / MAIN (unchanged semantics)  ===================== */

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
    char mode[16];
    mode[0] = '\0';
    strncpy(mode, "synthetic", sizeof(mode) - 1);
    char *filepath = NULL;
    int P = 4;
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

    int opt, idx, confidence;
    confidence = rand() % 6;
    while ((opt = getopt_long(argc, argv, "e:m:f:t:n:d:i:a:s:ph", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'e': confidence = atoi(optarg); break;
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

    double *pr = (double*)xmalloc((size_t)G.N * sizeof(double));
    double *pr_next = (double*)xmalloc((size_t)G.N * sizeof(double));

    for (int v = 0; v < G.N; v++) {
        pr[v] = 1.0 / (double)G.N;
        pr_next[v] = 0.0;
    }

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
        args[t].state = confidence;
        if (pthread_create(&threads[t], NULL, pagerank_worker, &args[t]) != 0) {
            die("pthread_create");
        }
    }

    double t0 = now_sec();
    for (int t = 0; t < P; t++) {
        pthread_join(threads[t], NULL);
    }
    double t1 = now_sec();

    printf("Time: %.6f seconds\n", t1 - t0);

    if (do_print) {
        PageRankEntry* ranked_pages = (PageRankEntry*)xmalloc((size_t)G.N * sizeof(PageRankEntry));
        double sum_ranks = 0.0;
        double min_rank = 1.0, max_rank = 0.0;
        
        for (int v = 0; v < G.N; v++) {
            ranked_pages[v].id = v;
            ranked_pages[v].rank = pr[v];
            sum_ranks += pr[v];
            if (pr[v] < min_rank) min_rank = pr[v];
            if (pr[v] > max_rank) max_rank = pr[v];
            
        }

        qsort(ranked_pages, (size_t)G.N, sizeof(PageRankEntry), compare_pagerank);

        printf("\n--- PageRank Statistics ---\n");
        printf("  Total Nodes: %d\n", G.N);
        printf("   Sum of Ranks: %.6f (should be ~1.0)\n", sum_ranks);
        printf("Average Rank: %.12f\n", sum_ranks / G.N);
        printf("    Min Rank: %.12f\n", min_rank);
        printf("    Max Rank: %.12f\n", max_rank);

        printf("\n--- Top 20 Ranked Pages ---\n");
        printf("Rank |   Node ID |   PageRank Score\n");
        printf("-----|-----------|------------------\n");
        int limit = (G.N < 20) ? G.N : 20;
        for (int i = 0; i < limit; i++) {

            printf("%4d | %9d | %.12f\n", i + 1, ranked_pages[i].id, ranked_pages[i].rank);
        }

        for (int v = 0; v < G.N; v++) {
            printf("pr(%d) = %.12f\n", v, pr[v]);
        }
        free(ranked_pages);
    }

    

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
