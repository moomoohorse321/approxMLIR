// RUN: cgeist -O0 %stdinclude %s -S > %s.mlir
// RUN: cgeist -O0 %stdinclude %s -o -lm %s.exec

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>


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


// -------------------- utilities (no knobs) --------------------
static int isInteger(const char *s){ if(!s||!*s) return 0; for(;*s;++s){ if(*s<'0'||*s>'9') return 0; } return 1; }

int approx_state_identity(int state) { return state; }

// -------------------- pair interaction (func_substitute knob) --------------------


int pair_interaction(int pi, int pj, fp a2,
                             FOUR_VECTOR* rv, fp* qv,
                             FOUR_VECTOR* fv_particle){
    double r2 = rv[pi].v + rv[pj].v - DOT(rv[pi], rv[pj]);
    if (r2 < 0) r2 = 0;
    double u2  = a2 * r2;
    double vij = exp(-u2);
    double fs  = 2.0 * vij;

    fv_particle->v += qv[pj] * vij;
    fv_particle->x += qv[pj] * (fs * (rv[pi].x - rv[pj].x));
    fv_particle->y += qv[pj] * (fs * (rv[pi].y - rv[pj].y));
    fv_particle->z += qv[pj] * (fs * (rv[pi].z - rv[pj].z));
    return 0;
}
// -------------------- self-box accumulate (loop_perforate knob) --------------------
// @approx:decision_tree {
//   transform_type: loop_perforate
//   state_indices: [6]
//   state_function: approx_state_identity
//   thresholds: [60]
//   thresholds_lower: [1]
//   thresholds_upper: [100]
//   decisions: [1, 2]
//   decision_values: [1, 2, 3, 4]
// }
static void self_box_accumulate(int pi_idx, int first_i, fp a2,
                                FOUR_VECTOR* rv, fp* qv,
                                FOUR_VECTOR* fv_particle, int state){
    int j;
    for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
        int pj_idx = first_i + j;
        if (pi_idx == pj_idx) continue;

        // caller-side state for nested knob A (based on u²)
        fp r2 = rv[pi_idx].v + rv[pj_idx].v - DOT(rv[pi_idx], rv[pj_idx]);
        if (r2 < (fp)0) r2 = (fp)0;
        fp u2 = a2 * r2;

        pair_interaction(pi_idx, pj_idx, a2, rv, qv, fv_particle);
    }
}

// -------------------- neighbor-box accumulate (loop_perforate knob) --------------------
// @approx:decision_tree {
//   transform_type: func_substitute
//   state_indices: [7]
//   state_function: approx_state_identity
//   thresholds: [6]
//   thresholds_lower: [1]
//   thresholds_upper: [10]
//   decisions: [1, 2]
//   decision_values: [0, 1, 2, 3]
// }
void neighbor_box_accumulate(int pi_idx, int bx, fp a2,
                                    box_str* box, FOUR_VECTOR* rv, fp* qv,
                                    FOUR_VECTOR* fv_particle, int state){
    int k;
    for (k = 0; k < box[bx].nn; k++) {
        int nb  = box[bx].nei[k].number;
        int off = box[nb].offset;

        int j;
        for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            int pj_idx = off + j;

            // caller-side state for nested knob A (based on u²)
            fp r2 = rv[pi_idx].v + rv[pj_idx].v - DOT(rv[pi_idx], rv[pj_idx]);
            if (r2 < (fp)0) r2 = (fp)0;
            fp u2 = a2 * r2;

            pair_interaction(pi_idx, pj_idx, a2, rv, qv, fv_particle);
        }
    }
}

void approx_neighbor_box_accumulate_1(int pi_idx, int bx, fp a2,
                                    box_str* box, FOUR_VECTOR* rv, fp* qv,
                                    FOUR_VECTOR* fv_particle, int state){
    int k;
    for (k = 0; k < box[bx].nn; k++) {
        int nb  = box[bx].nei[k].number;
        int off = box[nb].offset;

        int j;
        for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            if((j & 7) == 7) continue;
            int pj_idx = off + j;

            // caller-side state for nested knob A (based on u²)
            fp r2 = rv[pi_idx].v + rv[pj_idx].v - DOT(rv[pi_idx], rv[pj_idx]);
            if (r2 < (fp)0) r2 = (fp)0;
            fp u2 = a2 * r2;

            pair_interaction(pi_idx, pj_idx, a2, rv, qv, fv_particle);
        }
    }
}

void approx_neighbor_box_accumulate_2(int pi_idx, int bx, fp a2,
                                    box_str* box, FOUR_VECTOR* rv, fp* qv,
                                    FOUR_VECTOR* fv_particle, int state){
    int k;
    for (k = 0; k < box[bx].nn; k++) {
        int nb  = box[bx].nei[k].number;
        int off = box[nb].offset;

        int j;

        for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            int pj_idx = off + j;
            if((j & 3) == 3) continue;
            // caller-side state for nested knob A (based on u²)
            fp r2 = rv[pi_idx].v + rv[pj_idx].v - DOT(rv[pi_idx], rv[pj_idx]);
            if (r2 < (fp)0) r2 = (fp)0;
            fp u2 = a2 * r2;

            pair_interaction(pi_idx, pj_idx, a2, rv, qv, fv_particle);
        }
    }
}

void approx_neighbor_box_accumulate_3(int pi_idx, int bx, fp a2,
                                    box_str* box, FOUR_VECTOR* rv, fp* qv,
                                    FOUR_VECTOR* fv_particle, int state){
    int k;
    for (k = 0; k < box[bx].nn; k++) {
        int nb  = box[bx].nei[k].number;
        int off = box[nb].offset;

        int j;
        for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            if(j & 1) continue;
            int pj_idx = off + j;

            // caller-side state for nested knob A (based on u²)
            fp r2 = rv[pi_idx].v + rv[pj_idx].v - DOT(rv[pi_idx], rv[pj_idx]);
            if (r2 < (fp)0) r2 = (fp)0;
            fp u2 = a2 * r2;

            pair_interaction(pi_idx, pj_idx, a2, rv, qv, fv_particle);
        }
    }
}

// -------------------- non-knob wrapper over a box --------------------
static void process_home_box(int bx, fp a2, box_str* box,
                             FOUR_VECTOR* rv, fp* qv, FOUR_VECTOR* fv){
    int first_i = box[bx].offset;
    int i;
    for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {
        int pi_idx = first_i + i;

        FOUR_VECTOR acc = {0,0,0,0};

        int self_state = (int)(fabs(qv[pi_idx]) * 100.0);
        int neighbor_state = box[bx].nn;
        self_box_accumulate   (pi_idx, first_i, a2, rv, qv, &acc, self_state);
        neighbor_box_accumulate(pi_idx, bx,      a2, box, rv, qv, &acc, neighbor_state);

        fv[pi_idx].v += acc.v;
        fv[pi_idx].x += acc.x;
        fv[pi_idx].y += acc.y;
        fv[pi_idx].z += acc.z;
    }
}

// -------------------- main (unchanged I/O + build) --------------------
int main(int argc, char *argv[]) {
    par_str par_cpu;
    dim_str dim_cpu;
    box_str*     box_cpu = NULL;
    FOUR_VECTOR* rv_cpu  = NULL;
    fp*          qv_cpu  = NULL;
    FOUR_VECTOR* fv_cpu  = NULL;
    int nh;

    printf("WG size of kernel = %d\n", NUMBER_THREADS);

    dim_cpu.boxes1d_arg = 1;
    int seed = 2;
    if (argc == 3) {
        if (strcmp(argv[1], "-boxes1d") == 0 && isInteger(argv[2])) {
            dim_cpu.boxes1d_arg = atoi(argv[2]);
            if (dim_cpu.boxes1d_arg <= 0) { printf("ERROR: -boxes1d > 0\n"); return 1; }
        } else { printf("ERROR: Usage: %s -boxes1d <number>\n", argv[0]); return 1; }
    } else if (argc == 4) {
        if (strcmp(argv[1], "-boxes1d") == 0 && isInteger(argv[2])) {
            dim_cpu.boxes1d_arg = atoi(argv[2]);
            seed= atoi(argv[3]);
            if (dim_cpu.boxes1d_arg <= 0) { printf("ERROR: -boxes1d > 0\n"); return 1; }
        } else { printf("ERROR: Usage: %s -boxes1d <number>\n", argv[0]); return 1; }
    }else { printf("Usage: %s -boxes1d <number>\n", argv[0]); return 1; }

    printf("Configuration: boxes1d = %d\n", dim_cpu.boxes1d_arg);

    par_cpu.alpha        = (fp)0.5;
    dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;
    dim_cpu.space_elem   = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;

    box_cpu = (box_str*)malloc((size_t)dim_cpu.number_boxes * sizeof(box_str));
    rv_cpu  = (FOUR_VECTOR*)malloc((size_t)dim_cpu.space_elem   * sizeof(FOUR_VECTOR));
    qv_cpu  = (fp*)malloc((size_t)dim_cpu.space_elem            * sizeof(fp));
    fv_cpu  = (FOUR_VECTOR*)malloc((size_t)dim_cpu.space_elem   * sizeof(FOUR_VECTOR));
    if (!box_cpu || !rv_cpu || !qv_cpu || !fv_cpu) { printf("ERROR: OOM\n"); free(rv_cpu); free(qv_cpu); free(fv_cpu); free(box_cpu); return 1; }

    // build neighbors
    nh = 0;
    for (int i=0;i<dim_cpu.boxes1d_arg;i++){
      for (int j=0;j<dim_cpu.boxes1d_arg;j++){
        for (int k=0;k<dim_cpu.boxes1d_arg;k++){
          box_cpu[nh].number = nh;
          box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;
          box_cpu[nh].nn = 0;
          for (int l=-1;l<=1;l++) for (int m=-1;m<=1;m++) for (int n=-1;n<=1;n++){
            if (!(l==0 && m==0 && n==0)){
              int ni=i+l,nj=j+m,nk=k+n;
              if (ni>=0 && ni<dim_cpu.boxes1d_arg && nj>=0 && nj<dim_cpu.boxes1d_arg && nk>=0 && nk<dim_cpu.boxes1d_arg){
                int idx = box_cpu[nh].nn++;
                int num = (ni*dim_cpu.boxes1d_arg*dim_cpu.boxes1d_arg) + (nj*dim_cpu.boxes1d_arg) + nk;
                box_cpu[nh].nei[idx].number = num;
                box_cpu[nh].nei[idx].offset = num * NUMBER_PAR_PER_BOX;
              }
            }
          }
          nh++;
        }
      }
    }

    // init fields
    srand(seed);
    for (size_t i=0;i<(size_t)dim_cpu.space_elem;i++){
      rv_cpu[i].v=(fp)((rand()%10+1)/10.0);
      rv_cpu[i].x=(fp)((rand()%10+1)/10.0);
      rv_cpu[i].y=(fp)((rand()%10+1)/10.0);
      rv_cpu[i].z=(fp)((rand()%10+1)/10.0);
      qv_cpu[i]  =(fp)((rand()%10+1)/10.0);
      fv_cpu[i].v=fv_cpu[i].x=fv_cpu[i].y=fv_cpu[i].z=(fp)0.0;
    }

    // run
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    fp a2 = (fp)2.0 * par_cpu.alpha * par_cpu.alpha;
    for (int bx=0; bx<dim_cpu.number_boxes; ++bx) {
        process_home_box(bx, a2, box_cpu, rv_cpu, qv_cpu, fv_cpu);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1.0e6;
    printf("Total execution time: %.3f ms\n", elapsed_ms);

    printf("\n--- Simulation Statistics ---\n");
    printf("Total Particles: %ld\n", dim_cpu.space_elem);
    printf("Number of Boxes: %ld\n", dim_cpu.number_boxes);
    printf("Particles per Box: %d\n", NUMBER_PAR_PER_BOX);

    fp v_sum = 0.0, x_sum = 0.0, y_sum = 0.0, z_sum = 0.0;
    fp v_max = -8888888888.0, x_max = -8888888888.0, y_max = -8888888888.0, z_max = -8888888888.0;
    fp v_min = 8888888888.0,  x_min = 8888888888.0,  y_min = 8888888888.0,  z_min = 8888888888.0;

    for (size_t i = 0; i < (size_t)dim_cpu.space_elem; i++) {
        v_sum += fv_cpu[i].v;
        x_sum += fv_cpu[i].x;
        y_sum += fv_cpu[i].y;
        z_sum += fv_cpu[i].z;

        if (fv_cpu[i].v > v_max) v_max = fv_cpu[i].v;
        if (fv_cpu[i].x > x_max) x_max = fv_cpu[i].x;
        if (fv_cpu[i].y > y_max) y_max = fv_cpu[i].y;
        if (fv_cpu[i].z > z_max) z_max = fv_cpu[i].z;
        
        if (fv_cpu[i].v < v_min) v_min = fv_cpu[i].v;
        if (fv_cpu[i].x < x_min) x_min = fv_cpu[i].x;
        if (fv_cpu[i].y < y_min) y_min = fv_cpu[i].y;
        if (fv_cpu[i].z < z_min) z_min = fv_cpu[i].z;
    }

    printf("\n--- Result Summary (fv_cpu) ---\n");
    printf("        Component |      Average |          Min |          Max\n");
    printf("------------------|--------------|--------------|--------------\n");
    printf("Potential Energy (v) | %12.6f | %12.6f | %12.6f\n", v_sum / dim_cpu.space_elem, v_min, v_max);
    printf("      Force Vector (x) | %12.6f | %12.6f | %12.6f\n", x_sum / dim_cpu.space_elem, x_min, x_max);
    printf("      Force Vector (y) | %12.6f | %12.6f | %12.6f\n", y_sum / dim_cpu.space_elem, y_min, y_max);
    printf("      Force Vector (z) | %12.6f | %12.6f | %12.6f\n", z_sum / dim_cpu.space_elem, z_min, z_max);


    // Print the full results to stdout instead of writing to a file
    printf("\n--- Full Particle Data Dump (fv_cpu) ---\n");
    printf("Particle Index | Potential (v) |   Force (x) |   Force (y) |   Force (z)\n");
    printf("---------------|---------------|-------------|-------------|-------------\n");
    for (size_t i = 0; i < (size_t)dim_cpu.space_elem; i++) {
        printf("%14zu | %13.6f | %11.6f | %11.6f | %11.6f\n",
               i, fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
    }

        // write results
    FILE* fp_out = fopen("result.txt","wb");
    if (fp_out){
      size_t wrote = fwrite(&dim_cpu.space_elem, sizeof(dim_cpu.space_elem), 1, fp_out);
      if (wrote!=1) printf("ERROR: header write\n");
      wrote = fwrite(fv_cpu, sizeof(FOUR_VECTOR), (size_t)dim_cpu.space_elem, fp_out);
      if (wrote!=(size_t)dim_cpu.space_elem) printf("ERROR: data write\n");
      fclose(fp_out);
      printf("Results written to result.txt\n");
    } else { printf("ERROR: open result.txt\n"); }


    free(rv_cpu); free(qv_cpu); free(fv_cpu); free(box_cpu);
    return 0;
}
