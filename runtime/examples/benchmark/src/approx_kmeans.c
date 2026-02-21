#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

int seed = 42;

int approx_state_identity(int state) { return state; }

/* -------------------- Prototypes -------------------- */

double compute_distance_sq(const double *point1, const double *point2, int dim, int state);

int choose_cluster(const double *point, double **centroids, int k, int dim,
                   int dist_state, int state);

void reset_accumulators(double **new_centroids, int *cluster_sizes, int k, int dim);
void recompute_centroids(double **centroids, double **new_centroids,
                         const int *cluster_sizes, int k, int dim);

void kmeans_kernel(double **points, double **centroids, int *assignments,
                   int num_points, int dim, int k, int max_iters);

void generate_random_data(double **points, double **centroids, int num_points, int dim, int k);
void print_results(double **points, double **centroids, int *assignments,
                   int num_points, int dim, int k);

double compute_distance_sq(const double *point1, const double *point2, int dim, int state) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sum;
}

/* 2) Choose nearest centroid (knob: loop_perforate over centroids) */
// @approx:decision_tree {
//   transform_type: loop_perforate
//   state_indices: [5]
//   state_function: approx_state_identity
//   thresholds: [8]
//   thresholds_lower: [1]
//   thresholds_upper: [20]
//   decisions: [0, 0]
//   decision_values: [0, 1, 2, 3, 4]
// }
int choose_cluster(const double *point, double **centroids, int k, int dim,
                   int dist_state, int state) {
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

// @approx:decision_tree {
//   transform_type: loop_perforate
//   state_indices: [9]
//   state_function: approx_state_identity
//   thresholds: [10000]
//   thresholds_lower: [1]
//   thresholds_upper: [20]
//   decisions: [0, 0]
//   decision_values: [0, 1, 2, 3, 4]
// }
void assign_points_and_accumulate(double **points, double **centroids, int *assignments,
                                  int *cluster_sizes, double **new_centroids,
                                  int num_points, int dim, int k,
                                  int choose_state, int state) {
    for (int i = 0; i < num_points; i++) {
        int best = choose_cluster(points[i], centroids, k, dim, i, choose_state);
        assignments[i] = best;
        cluster_sizes[best]++;
        for (int j = 0; j < dim; j++) {
            new_centroids[best][j] += points[i][j];
        }
    }
}

void recompute_centroids(double **centroids, double **new_centroids,
                         const int *cluster_sizes, int k, int dim) {
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

// @approx:decision_tree {
//   transform_type: loop_perforate
//   state_indices: [9]
//   state_function: approx_state_identity
//   thresholds: [10000]
//   thresholds_lower: [1]
//   thresholds_upper: [1000000]
//   decisions: [0, 0]
//   decision_values: [0, 1, 2, 3, 4]
// }
void run_kmeans_iterations(int max_iters, int k, int dim, int num_points,
                           double **points, double **centroids, int *assignments,
                           double **new_centroids, int *cluster_sizes, int state) {
    for (int iter = 0; iter < max_iters; iter++) {
        reset_accumulators(new_centroids, cluster_sizes, k, dim);
        assign_points_and_accumulate(points, centroids, assignments,
                                     cluster_sizes, new_centroids,
                                     num_points, dim, k,
                                     iter /* choose state*/, iter /* pt_state */);
        recompute_centroids(centroids, new_centroids, cluster_sizes, k, dim);
    }
}

void kmeans_kernel(double **points, double **centroids, int *assignments,
                   int num_points, int dim, int k, int max_iters) {
    // Allocate memory for temporary data
    int *cluster_sizes = (int *)malloc((size_t)k * sizeof(int));
    double **new_centroids = (double **)malloc((size_t)k * sizeof(double *));
    for (int c = 0; c < k; c++) {
        new_centroids[c] = (double *)malloc((size_t)dim * sizeof(double));
    }

    // Run the main K-Means iterations
    run_kmeans_iterations(max_iters, k, dim, num_points, points, centroids,
                          assignments, new_centroids, cluster_sizes, num_points);

    // Free the allocated memory
    for (int c = 0; c < k; c++) {
        free(new_centroids[c]);
    }
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

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    kmeans_kernel(points, centroids, assignments, num_points, dim, k, max_iters);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                       (end.tv_nsec - start.tv_nsec) / 1.0e6;
    printf("\nK-means completed in %.3f ms\n", elapsed_ms);

    print_results(points, centroids, assignments, num_points, dim, k);

    for (int i = 0; i < num_points; i++) free(points[i]);
    free(points);
    for (int i = 0; i < k; i++) free(centroids[i]);
    free(centroids);
    free(assignments);
    return 0;
}
