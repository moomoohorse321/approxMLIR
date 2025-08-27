#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <fstream>
#include <math.h>
#include <vector>
#include <sys/time.h> // For gettimeofday()

// All structs, types, and constants are now defined in main.h
#include "main.h"

//====================================================================================================
//  UTILITY FUNCTIONS
//====================================================================================================

// Returns the current system time in microseconds
long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

// Checks if a string represents an integer
int isInteger(const char *str){
    if (*str == '\0') return 0;
    for(; *str != '\0'; str++){
        if (*str < '0' || *str > '9') return 0;
    }
    return 1;
}

void calculate_particle_interaction_precise(int p_i_idx, int p_j_idx, fp a2, FOUR_VECTOR* rv_cpu, fp* qv_cpu, FOUR_VECTOR* fv_particle) {
    fp r2 = rv_cpu[p_i_idx].v + rv_cpu[p_j_idx].v - DOT(rv_cpu[p_i_idx], rv_cpu[p_j_idx]);
    fp u2 = a2 * r2;
    fp vij = exp(-u2); // Precise, expensive calculation
    fp fs = 2 * vij;
    
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

void calculate_self_box_interactions(int p_i_idx, int first_i, fp a2, FOUR_VECTOR* rv_cpu, fp* qv_cpu, FOUR_VECTOR* fv_particle) {
    for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
        if (p_i_idx == first_i + j) continue;
        // This call is the target for function substitution
        calculate_particle_interaction_precise(p_i_idx, first_i + j, a2, rv_cpu, qv_cpu, fv_particle);
    }
}

void calculate_neighbor_box_interactions(int p_i_idx, int bx, fp a2, box_str* box_cpu, FOUR_VECTOR* rv_cpu, fp* qv_cpu, FOUR_VECTOR* fv_particle) {
    for (int k = 0; k < box_cpu[bx].nn; k++) {
        int pointer = box_cpu[bx].nei[k].number;
        int first_j = box_cpu[pointer].offset;
        for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
            // This call is the target for function substitution
            calculate_particle_interaction_precise(p_i_idx, first_j + j, a2, rv_cpu, qv_cpu, fv_particle);
        }
    }
}

void process_home_box(int bx, fp a2, box_str* box_cpu, FOUR_VECTOR* rv_cpu, fp* qv_cpu, FOUR_VECTOR* fv_cpu) {
    int first_i = box_cpu[bx].offset;
    for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {
        int p_i_idx = first_i + i;
        FOUR_VECTOR fv_particle = {0.0, 0.0, 0.0, 0.0};
        calculate_self_box_interactions(p_i_idx, first_i, a2, rv_cpu, qv_cpu, &fv_particle);
        calculate_neighbor_box_interactions(p_i_idx, bx, a2, box_cpu, rv_cpu, qv_cpu, &fv_particle);
        fv_cpu[p_i_idx].v += fv_particle.v;
        fv_cpu[p_i_idx].x += fv_particle.x;
        fv_cpu[p_i_idx].y += fv_particle.y;
        fv_cpu[p_i_idx].z += fv_particle.z;
    }
}


int main(int argc, char *argv[]) {
    par_str par_cpu;
    dim_str dim_cpu;
    box_str* box_cpu;
    FOUR_VECTOR* rv_cpu;
    fp* qv_cpu;
    FOUR_VECTOR* fv_cpu;
    int nh;

    printf("WG size of kernel = %d \n", NUMBER_THREADS);

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
    par_cpu.alpha = 0.5;
    dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;
    dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
    
    box_cpu = (box_str*)malloc(dim_cpu.number_boxes * sizeof(box_str));
    rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_elem * sizeof(FOUR_VECTOR));
    qv_cpu = (fp*)malloc(dim_cpu.space_elem * sizeof(fp));
    fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_elem * sizeof(FOUR_VECTOR));

    if (!box_cpu || !rv_cpu || !qv_cpu || !fv_cpu) {
        printf("ERROR: Memory allocation failed\n");
        return 1;
    }

    nh = 0;
    for (int i = 0; i < dim_cpu.boxes1d_arg; i++) {
        for (int j = 0; j < dim_cpu.boxes1d_arg; j++) {
            for (int k = 0; k < dim_cpu.boxes1d_arg; k++) {
                box_cpu[nh].number = nh;
                box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;
                box_cpu[nh].nn = 0;
                for (int l = -1; l <= 1; l++) {
                    for (int m = -1; m <= 1; m++) {
                        for (int n = -1; n <= 1; n++) {
                            if (!(l == 0 && m == 0 && n == 0)) {
                                int ni = i + l, nj = j + m, nk = k + n;
                                if (ni >= 0 && ni < dim_cpu.boxes1d_arg && nj >= 0 && nj < dim_cpu.boxes1d_arg && nk >= 0 && nk < dim_cpu.boxes1d_arg) {
                                    int neighbor_idx = box_cpu[nh].nn++;
                                    box_cpu[nh].nei[neighbor_idx].number = (ni * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + (nj * dim_cpu.boxes1d_arg) + nk;
                                    box_cpu[nh].nei[neighbor_idx].offset = box_cpu[nh].nei[neighbor_idx].number * NUMBER_PAR_PER_BOX;
                                }
                            }
                        }
                    }
                }
                nh++;
            }
        }
    }

    srand(2);
    for (int i = 0; i < dim_cpu.space_elem; i++) {
        rv_cpu[i].v = (rand() % 10 + 1) / 10.0;
        rv_cpu[i].x = (rand() % 10 + 1) / 10.0;
        rv_cpu[i].y = (rand() % 10 + 1) / 10.0;
        rv_cpu[i].z = (rand() % 10 + 1) / 10.0;
        qv_cpu[i] = (rand() % 10 + 1) / 10.0;
        fv_cpu[i].v = 0.0;
        fv_cpu[i].x = 0.0;
        fv_cpu[i].y = 0.0;
        fv_cpu[i].z = 0.0;
    }

    long long start_time = get_time();
    fp a2 = 2 * par_cpu.alpha * par_cpu.alpha;
    for (int bx = 0; bx < dim_cpu.number_boxes; bx++) {
        process_home_box(bx, a2, box_cpu, rv_cpu, qv_cpu, fv_cpu);
    }
    long long end_time = get_time();
    
    printf("Total execution time: %f seconds\n", (end_time - start_time) / 1000000.0);

    // File output is now enabled by default
    std::ofstream myFile("result.txt", std::ios::out | std::ios::binary);
    if (myFile.is_open()) {
        myFile.write(reinterpret_cast<char*>(&dim_cpu.space_elem), sizeof(dim_cpu.space_elem));
        myFile.write(reinterpret_cast<char*>(fv_cpu), dim_cpu.space_elem * sizeof(FOUR_VECTOR));
        myFile.close();
        printf("Results written to result.txt\n");
    } else {
        printf("ERROR: Could not open result.txt for writing\n");
    }

    free(rv_cpu);
    free(qv_cpu);
    free(fv_cpu);
    free(box_cpu);

    return 0;
}
