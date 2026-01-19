// RUN: cgeist -O0 %stdinclude %s -S > %s.mlir
// RUN: cgeist -O0 %stdinclude %s -o %s.exec

// // Knob — distance computation (func_substitute)
// "approxMLIR.util.annotation.decision_tree"() <{
//   func_name = "compute_distance_sq",
//   transform_type = "func_substitute",
//   num_thresholds = 1 : i32,
//   thresholds_uppers = array<i32: 64>,
//   thresholds_lowers = array<i32: 0>,
//   decision_values = array<i32: 0, 1>,
//   thresholds = array<i32: 16>,
//   decisions = array<i32: 0, 1>
// }> : () -> ()
// // Required for func_substitute
// "approxMLIR.util.annotation.convert_to_call"() <{func_name = "compute_distance_sq"}> : () -> ()

// // Knob — choosing nearest centroid (loop_perforate over centroids)
// "approxMLIR.util.annotation.decision_tree"() <{
//   func_name = "choose_cluster",
//   transform_type = "loop_perforate",
//   num_thresholds = 1 : i32,
//   thresholds_uppers = array<i32: 16>,
//   thresholds_lowers = array<i32: 0>,
//   decision_values = array<i32: 0, 1>,
//   thresholds = array<i32: 8>,
//   decisions = array<i32: 1, 2>
// }> : () -> ()

// // Knob — assigning points & accumulating (loop_perforate over points)
// "approxMLIR.util.annotation.decision_tree"() <{
//   func_name = "assign_points_and_accumulate",
//   transform_type = "loop_perforate",
//   num_thresholds = 1 : i32,
//   thresholds_uppers = array<i32: 2000000>,
//   thresholds_lowers = array<i32: 0>,
//   decision_values = array<i32: 0, 1>,
//   thresholds = array<i32: 10000>,
//   decisions = array<i32: 1, 2>
// }> : () -> ()

#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

int seed = 42;

/* -------------------- Prototypes -------------------- */
static inline int decide_distance_state(int dim);
static inline int decide_choose_state(int k);
static inline int decide_points_state(int num_points);

double compute_distance_sq(const double *point1, const double *point2, int dim, int state);
double approx_compute_distance_sq(const double *point1, const double *point2, int dim, int state);

int choose_cluster(const double *point, double **centroids, int k, int dim,
                   int dist_state, int state);

void reset_accumulators(double **new_centroids, int *cluster_sizes, int k, int dim);
void assign_points_and_accumulate(double **points, double **centroids, int *assignments,
                                  int *cluster_sizes, double **new_centroids,
                                  int num_points, int dim, int k,
                                  int dist_state, int choose_state, int state);
void recompute_centroids(double **centroids, double **new_centroids,
                         const int *cluster_sizes, int k, int dim, int state);

void kmeans_kernel(double **points, double **centroids, int *assignments,
                   int num_points, int dim, int k, int max_iters);

void generate_random_data(double **points, double **centroids, int num_points, int dim, int k);
void print_results(double **points, double **centroids, int *assignments,
                   int num_points, int dim, int k);

/* -------------------- State deciders (caller side) -------------------- */
static inline int decide_distance_state(int dim) {
    /* Heuristic: higher dim → allow stronger approx (state=1), else exact (0) */
    return (dim > 16) ? 1 : 0;
}
static inline int decide_choose_state(int k) {
    /* Heuristic: many clusters → allow perforation (state=1) */
    return (k > 8) ? 1 : 0;
}
static inline int decide_points_state(int num_points) {
    /* Heuristic: many points → allow perforation (state=1) */
    return (num_points > 10000) ? 1 : 0;
}

/* -------------------- Compute-heavy kernels (ONE KNOB EACH) -------------------- */
/* 1) Exact distance squared (knob: func_substitute on this function) */
double compute_distance_sq(const double *point1, const double *point2, int dim, int state) {
    (void)state; /* exact baseline ignores state */
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sum;
}

/* Approximate variant for func_substitute (not called in exact run) */
double approx_compute_distance_sq(const double *point1, const double *point2, int dim, int state) {
    double sum = 0.0;
    for (int i = 0; i < dim; i += 2) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    /* scale to roughly compensate */
    if (dim > 1) sum *= 2.0;
    return sum;
}

/* 2) Choose nearest centroid (knob: loop_perforate over centroids) */
int choose_cluster(const double *point, double **centroids, int k, int dim,
                   int dist_state, int state) {
    (void)state; /* exact baseline ignores state */
    double min_dist = DBL_MAX;
    int best_cluster = 0;
    for (int c = 0; c < k; c++) {
        double d = compute_distance_sq(point, centroids[c], dim, dist_state);
        if (d < min_dist) {
            min_dist = d;
            best_cluster = c;
        }
    }
    return best_cluster;
}

/* 3) Assign all points and accumulate sums (knob: loop_perforate over points) */
void reset_accumulators(double **new_centroids, int *cluster_sizes, int k, int dim) {
    for (int c = 0; c < k; c++) {
        cluster_sizes[c] = 0;
        for (int j = 0; j < dim; j++) new_centroids[c][j] = 0.0;
    }
}

void assign_points_and_accumulate(double **points, double **centroids, int *assignments,
                                  int *cluster_sizes, double **new_centroids,
                                  int num_points, int dim, int k,
                                  int dist_state, int choose_state, int state) {
    (void)state; /* exact baseline ignores state */
    for (int i = 0; i < num_points; i++) {
        int best = choose_cluster(points[i], centroids, k, dim, dist_state, choose_state);
        assignments[i] = best;
        cluster_sizes[best]++;
        for (int j = 0; j < dim; j++) {
            new_centroids[best][j] += points[i][j];
        }
    }
}

void recompute_centroids(double **centroids, double **new_centroids,
                         const int *cluster_sizes, int k, int dim, int state) {
    (void)state; /* exact baseline ignores state */
    for (int c = 0; c < k; c++) {
        if (cluster_sizes[c] > 0) {
            double inv = 1.0 / (double)cluster_sizes[c];
            for (int j = 0; j < dim; j++) {
                centroids[c][j] = new_centroids[c][j] * inv;
            }
        }
    }
}

/* -------------------- K-means driver (refactored to call kernels) -------------------- */
void kmeans_kernel(double **points, double **centroids, int *assignments,
                   int num_points, int dim, int k, int max_iters) {
    int c, iter, j;

    int *cluster_sizes = (int *)malloc((size_t)k * sizeof(int));
    double **new_centroids = (double **)malloc((size_t)k * sizeof(double *));
    for (c = 0; c < k; c++) {
        new_centroids[c] = (double *)malloc((size_t)dim * sizeof(double));
    }

    /* Decide states once per run (cheap heuristics; arbitrary) */
    int dist_state   = decide_distance_state(dim);
    int choose_state = decide_choose_state(k);
    int pts_state    = decide_points_state(num_points);

    for (iter = 0; iter < max_iters; iter++) {
        reset_accumulators(new_centroids, cluster_sizes, k, dim);
        assign_points_and_accumulate(points, centroids, assignments,
                                     cluster_sizes, new_centroids,
                                     num_points, dim, k,
                                     dist_state, choose_state, pts_state);
        recompute_centroids(centroids, new_centroids, cluster_sizes, k, dim, 0);
    }

    for (c = 0; c < k; c++) free(new_centroids[c]);
    free(new_centroids);
    free(cluster_sizes);
}

/* -------------------- Helpers (no knobs) -------------------- */
void generate_random_data(double **points, double **centroids, int num_points, int dim, int k) {
    srand((unsigned int)seed);
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < dim; j++) {
            points[i][j] = (double)rand() / (double)RAND_MAX * 100.0;
        }
    }
    for (int i = 0; i < k; i++) {
        int idx = rand() % num_points;
        for (int j = 0; j < dim; j++) {
            centroids[i][j] = points[idx][j];
        }
    }
}

void print_results(double **points, double **centroids, int *assignments, int num_points, int dim, int k) {
    printf("Final centroids:\n");
    for (int i = 0; i < k; i++) {
        printf("Centroid %d: (", i);
        for (int j = 0; j < dim; j++) {
            printf("%.2f", centroids[i][j]);
            if (j < dim - 1) printf(", ");
        }
        printf(")\n");
    }
    int *cluster_sizes = (int *)malloc((size_t)k * sizeof(int));
    for (int i = 0; i < k; i++) cluster_sizes[i] = 0;
    for (int i = 0; i < num_points; i++) cluster_sizes[assignments[i]]++;
    printf("\nCluster sizes:\n");
    for (int i = 0; i < k; i++) printf("Cluster %d: %d points\n", i, cluster_sizes[i]);
    printf("\nSample points from each cluster:\n");
    for (int c = 0; c < k; c++) {
        printf("Cluster %d samples:\n", c);
        int count = 0;
        for (int i = 0; i < num_points && count < 3; i++) {
            if (assignments[i] == c) {
                printf("  Point %d: (", i);
                for (int j = 0; j < dim; j++) {
                    printf("%.2f", points[i][j]);
                    if (j < dim - 1) printf(", ");
                }
                printf(")\n");
                count++;
            }
        }
    }
    free(cluster_sizes);
}

/* -------------------- Main -------------------- */
int main(int argc, char **argv) {
    int num_points = 1000;
    int dim = 2;
    int k = 5;
    int max_iters = 20;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i+1 < argc) { num_points = atoi(argv[i+1]); i++; }
        else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) { dim = atoi(argv[i+1]); i++; }
        else if (strcmp(argv[i], "-k") == 0 && i+1 < argc) { k = atoi(argv[i+1]); i++; }
        else if (strcmp(argv[i], "-i") == 0 && i+1 < argc) { max_iters = atoi(argv[i+1]); i++; }
        else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) { seed = atoi(argv[i+1]); i++; } 
        else if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [-n num_points] [-d dimensions] [-k clusters] [-i max_iterations]\n", argv[0]);
            return 0;
        }
    }

    printf("Running K-means with:\n");
    printf("  Points: %d\n", num_points);
    printf("  Dimensions: %d\n", dim);
    printf("  Clusters: %d\n", k);
    printf("  Max iterations: %d\n", max_iters);

    double **points = (double **)malloc((size_t)num_points * sizeof(double *));
    for (int i = 0; i < num_points; i++) points[i] = (double *)malloc((size_t)dim * sizeof(double));

    double **centroids = (double **)malloc((size_t)k * sizeof(double *));
    for (int i = 0; i < k; i++) centroids[i] = (double *)malloc((size_t)dim * sizeof(double));
 
    int *assignments = (int *)malloc((size_t)num_points * sizeof(int));

    generate_random_data(points, centroids, num_points, dim, k);

    clock_t start = clock();
    kmeans_kernel(points, centroids, assignments, num_points, dim, k, max_iters);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("\nK-means completed in %.3f seconds\n", time_spent);

    print_results(points, centroids, assignments, num_points, dim, k);

    for (int i = 0; i < num_points; i++) free(points[i]);
    free(points);
    for (int i = 0; i < k; i++) free(centroids[i]);
    free(centroids);
    free(assignments);
    return 0;
}
