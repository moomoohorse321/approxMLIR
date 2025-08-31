// pure_c_main.c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>   // gettimeofday

//====================================================================================================
//  UTILITY FUNCTIONS
//====================================================================================================

#define fp double

#define NUMBER_PAR_PER_BOX 128							// keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

//#define NUMBER_THREADS 128								// this should be roughly equal to NUMBER_PAR_PER_BOX for best performance
// Parameterized work group size
#ifdef RD_WG_SIZE_0_0
        #define NUMBER_THREADS RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define NUMBER_THREADS RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define NUMBER_THREADS RD_WG_SIZE
#else
        #define NUMBER_THREADS 128
#endif

#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))	// STABLE


typedef struct
{
	fp x, y, z;

} THREE_VECTOR;

typedef struct
{
	fp v, x, y, z;

} FOUR_VECTOR;

typedef struct nei_str
{

	// neighbor box
	int x, y, z;
	int number;
	long offset;

} nei_str;

typedef struct box_str
{

	// home box
	int x, y, z;
	int number;
	long offset;

	// neighbor boxes
	int nn;
	nei_str nei[26];

} box_str;

typedef struct par_str
{

	fp alpha;

} par_str;

typedef struct dim_str
{

	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;

	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;
	long space_mem;
	long space_mem2;

} dim_str;


// Returns the current system time in microseconds
static long long get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000000LL + (long long)tv.tv_usec;
}

// Checks if a string represents a non-negative integer
static int isInteger(const char *str){
    if (str == NULL || *str == '\0') return 0;
    for (; *str != '\0'; str++){
        if (*str < '0' || *str > '9') return 0;
    }
    return 1;
}

//====================================================================================================
//  PHYSICS KERNEL HELPERS
//====================================================================================================

static void calculate_particle_interaction_precise(int p_i_idx,
                                                   int p_j_idx,
                                                   fp a2,
                                                   FOUR_VECTOR* rv_cpu,
                                                   fp* qv_cpu,
                                                   FOUR_VECTOR* fv_particle)
{
    fp r2  = rv_cpu[p_i_idx].v + rv_cpu[p_j_idx].v - DOT(rv_cpu[p_i_idx], rv_cpu[p_j_idx]);
    fp u2  = a2 * r2;
    fp vij = (fp)exp(-u2);  // precise, more expensive
    fp fs  = (fp)2.0 * vij;

    THREE_VECTOR d;
    d.x = rv_cpu[p_i_idx].x - rv_cpu[p_j_idx].x;
    d.y = rv_cpu[p_i_idx].y - rv_cpu[p_j_idx].y;
    d.z = rv_cpu[p_i_idx].z - rv_cpu[p_j_idx].z;

    fp fxij = fs * d.x;
    fp fyij = fs * d.y;
    fp fzij = fs * d.z;

    fv_particle->v += qv_cpu[p_j_idx] * vij;
    fv_particle->x += qv_cpu[p_j_idx] * fxij;
    fv_particle->y += qv_cpu[p_j_idx] * fyij;
    fv_particle->z += qv_cpu[p_j_idx] * fzij;
}

static void calculate_self_box_interactions(int p_i_idx,
                                            int first_i,
                                            fp a2,
                                            FOUR_VECTOR* rv_cpu,
                                            fp* qv_cpu,
                                            FOUR_VECTOR* fv_particle)
{
    int j;
    for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
        if (p_i_idx == first_i + j) continue;
        // Target for function substitution if needed
        calculate_particle_interaction_precise(p_i_idx, first_i + j, a2, rv_cpu, qv_cpu, fv_particle);
    }
}

static void calculate_neighbor_box_interactions(int p_i_idx,
                                                int bx,
                                                fp a2,
                                                box_str* box_cpu,
                                                FOUR_VECTOR* rv_cpu,
                                                fp* qv_cpu,
                                                FOUR_VECTOR* fv_particle)
{
    int k;
    for (k = 0; k < box_cpu[bx].nn; k++) {
        int pointer = box_cpu[bx].nei[k].number;
        int first_j = box_cpu[pointer].offset;
        int j;
        for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            // Target for function substitution if needed
            calculate_particle_interaction_precise(p_i_idx, first_j + j, a2, rv_cpu, qv_cpu, fv_particle);
        }
    }
}

static void process_home_box(int bx,
                             fp a2,
                             box_str* box_cpu,
                             FOUR_VECTOR* rv_cpu,
                             fp* qv_cpu,
                             FOUR_VECTOR* fv_cpu)
{
    int first_i = box_cpu[bx].offset;
    int i;
    for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {
        int p_i_idx = first_i + i;
        FOUR_VECTOR fv_particle;
        fv_particle.v = 0.0;
        fv_particle.x = 0.0;
        fv_particle.y = 0.0;
        fv_particle.z = 0.0;

        calculate_self_box_interactions(p_i_idx, first_i, a2, rv_cpu, qv_cpu, &fv_particle);
        calculate_neighbor_box_interactions(p_i_idx, bx, a2, box_cpu, rv_cpu, qv_cpu, &fv_particle);

        fv_cpu[p_i_idx].v += fv_particle.v;
        fv_cpu[p_i_idx].x += fv_particle.x;
        fv_cpu[p_i_idx].y += fv_particle.y;
        fv_cpu[p_i_idx].z += fv_particle.z;
    }
}

//====================================================================================================
//  MAIN
//====================================================================================================

int main(int argc, char *argv[]) {
    par_str      par_cpu;
    dim_str      dim_cpu;
    box_str*     box_cpu = NULL;
    FOUR_VECTOR* rv_cpu  = NULL;
    fp*          qv_cpu  = NULL;
    FOUR_VECTOR* fv_cpu  = NULL;
    int nh;

    printf("WG size of kernel = %d\n", NUMBER_THREADS);

    dim_cpu.boxes1d_arg = 1;
    if (argc == 3) {
        if (strcmp(argv[1], "-boxes1d") == 0 && isInteger(argv[2])) {
            dim_cpu.boxes1d_arg = atoi(argv[2]);
            if (dim_cpu.boxes1d_arg <= 0) {
                printf("ERROR: -boxes1d argument must be > 0\n");
                return 1;
            }
        } else {
            printf("ERROR: Invalid arguments. Usage: %s -boxes1d <number>\n", argv[0]);
            return 1;
        }
    } else {
        printf("Usage: %s -boxes1d <number>\n", argv[0]);
        return 1;
    }

    printf("Configuration: boxes1d = %d\n", dim_cpu.boxes1d_arg);

    par_cpu.alpha       = (fp)0.5;
    dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;
    dim_cpu.space_elem   = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;

    box_cpu = (box_str*)malloc((size_t)dim_cpu.number_boxes * sizeof(box_str));
    rv_cpu  = (FOUR_VECTOR*)malloc((size_t)dim_cpu.space_elem   * sizeof(FOUR_VECTOR));
    qv_cpu  = (fp*)malloc((size_t)dim_cpu.space_elem            * sizeof(fp));
    fv_cpu  = (FOUR_VECTOR*)malloc((size_t)dim_cpu.space_elem   * sizeof(FOUR_VECTOR));

    if (!box_cpu || !rv_cpu || !qv_cpu || !fv_cpu) {
        printf("ERROR: Memory allocation failed\n");
        free(rv_cpu); free(qv_cpu); free(fv_cpu); free(box_cpu);
        return 1;
    }

    // Build grid & neighbor lists
    nh = 0;
    {
        int i, j, k;
        for (i = 0; i < dim_cpu.boxes1d_arg; i++) {
            for (j = 0; j < dim_cpu.boxes1d_arg; j++) {
                for (k = 0; k < dim_cpu.boxes1d_arg; k++) {
                    int l, m, n;
                    box_cpu[nh].number = nh;
                    box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;
                    box_cpu[nh].nn     = 0;

                    for (l = -1; l <= 1; l++) {
                        for (m = -1; m <= 1; m++) {
                            for (n = -1; n <= 1; n++) {
                                if (!(l == 0 && m == 0 && n == 0)) {
                                    int ni = i + l, nj = j + m, nk = k + n;
                                    if (ni >= 0 && ni < dim_cpu.boxes1d_arg &&
                                        nj >= 0 && nj < dim_cpu.boxes1d_arg &&
                                        nk >= 0 && nk < dim_cpu.boxes1d_arg)
                                    {
                                        int neighbor_idx = box_cpu[nh].nn++;
                                        int number = (ni * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) +
                                                     (nj * dim_cpu.boxes1d_arg) + nk;
                                        box_cpu[nh].nei[neighbor_idx].number = number;
                                        box_cpu[nh].nei[neighbor_idx].offset = number * NUMBER_PAR_PER_BOX;
                                    }
                                }
                            }
                        }
                    }
                    nh++;
                }
            }
        }
    }

    // Initialize data
    srand(2);
    {
        size_t i;
        for (i = 0; i < (size_t)dim_cpu.space_elem; i++) {
            rv_cpu[i].v = (fp)((rand() % 10 + 1) / 10.0);
            rv_cpu[i].x = (fp)((rand() % 10 + 1) / 10.0);
            rv_cpu[i].y = (fp)((rand() % 10 + 1) / 10.0);
            rv_cpu[i].z = (fp)((rand() % 10 + 1) / 10.0);
            qv_cpu[i]   = (fp)((rand() % 10 + 1) / 10.0);
            fv_cpu[i].v = (fp)0.0;
            fv_cpu[i].x = (fp)0.0;
            fv_cpu[i].y = (fp)0.0;
            fv_cpu[i].z = (fp)0.0;
        }
    }

    // Run
    {
        long long start_time = get_time();
        fp a2 = (fp)2.0 * par_cpu.alpha * par_cpu.alpha;

        int bx;
        for (bx = 0; bx < dim_cpu.number_boxes; bx++) {
            process_home_box(bx, a2, box_cpu, rv_cpu, qv_cpu, fv_cpu);
        }

        long long end_time = get_time();
        printf("Total execution time: %f seconds\n",
               (double)(end_time - start_time) / 1000000.0);
    }

    // Binary file output (C stdio)
    {
        FILE *fp_out = fopen("result.txt", "wb");
        if (fp_out) {
            size_t wrote;
            wrote = fwrite(&dim_cpu.space_elem, sizeof(dim_cpu.space_elem), 1, fp_out);
            if (wrote != 1) {
                printf("ERROR: Failed to write header to result.txt\n");
            }
            wrote = fwrite(fv_cpu, sizeof(FOUR_VECTOR), (size_t)dim_cpu.space_elem, fp_out);
            if (wrote != (size_t)dim_cpu.space_elem) {
                printf("ERROR: Failed to write data to result.txt\n");
            }
            fclose(fp_out);
            printf("Results written to result.txt\n");
        } else {
            printf("ERROR: Could not open result.txt for writing\n");
        }
    }

    free(rv_cpu);
    free(qv_cpu);
    free(fv_cpu);
    free(box_cpu);
    return 0;
}
