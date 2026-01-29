#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_DOCS 100000
#define EMBEDDING_DIM 384
#define MAX_LINE_LENGTH 100000

typedef struct {
    int id;
    char *embedding;
    char title[256];
    float similarity;
} Document;

Document documents[MAX_DOCS];
int num_documents = 0;
float query_embedding[EMBEDDING_DIM];

int approx_state_identity(int state) { return state; }


// @approx:decision_tree {
//   transform_type: func_substitute
//   state_indices: [2]
//   state_function: approx_state_identity
//   thresholds: [1]
//   thresholds_lower: [1]
//   thresholds_upper: [10]
//   decisions: [0, 1]
//   decision_values: [0, 1, 2]
// }
int parse_embedding(const char *embedding_str, float *embedding, int state) {
    const char *start = strchr(embedding_str, '[');
    if (!start) return 0;
    start++;

    int count = 0;
    char *copy = strdup(start);
    if (!copy) return 0;
    char *saveptr = NULL;

    char *token = strtok_r(copy, ",]", &saveptr);
    while (token && count < EMBEDDING_DIM) {
        embedding[count] = (float)atof(token);
        count++;
        token = strtok_r(NULL, ",]", &saveptr);
    }


    free(copy);
    return count == EMBEDDING_DIM;
}

int approx_parse_embedding_1(const char *embedding_str, float *embedding, int state) {
    const char *start = strchr(embedding_str, '[');
    if (!start) return 0;
    start++;

    int count = 0;
    char *copy = strdup(start);
    if (!copy) return 0;
    char *saveptr = NULL;

    char *token = strtok_r(copy, ",]", &saveptr);
    while (token && count < EMBEDDING_DIM) {
        token[8] = '\0';
        embedding[count] = (float)atof(token);
        count++;
        token = strtok_r(NULL, ",]", &saveptr);
    }


    free(copy);
    return count == EMBEDDING_DIM;
}

int approx_parse_embedding_2(const char *embedding_str, float *embedding, int state) {
    const char *start = strchr(embedding_str, '[');
    if (!start) return 0;
    start++;

    int count = 0;
    char *copy = strdup(start);
    if (!copy) return 0;
    char *saveptr = NULL;

    char *token = strtok_r(copy, ",]", &saveptr);
    while (token && count < EMBEDDING_DIM) {
        token[5] = '\0';
        embedding[count] = (float)atof(token);
        count++;
        token = strtok_r(NULL, ",]", &saveptr);
    }


    free(copy);
    return count == EMBEDDING_DIM;
}

int load_document_embeddings(void) {
    char line[MAX_LINE_LENGTH];
    num_documents = 0;

    while (fgets(line, sizeof(line), stdin) && num_documents < MAX_DOCS) {
        if (strlen(line) <= 1) continue;

        char *doc_id_str = strtok(line, "|");
        char *title = strtok(NULL, "|");
        char *embedding_str = strtok(NULL, "\n");

        if (!doc_id_str || !title || !embedding_str) continue;

        documents[num_documents].id = atoi(doc_id_str);
        strncpy(documents[num_documents].title, title, 255);
        documents[num_documents].title[255] = '\0';

        // MODIFIED: Instead of parsing, we now allocate memory and copy the embedding string.
        documents[num_documents].embedding = strdup(embedding_str);
        if (documents[num_documents].embedding) {
            num_documents++;
        } else {
            fprintf(stderr, "Error: Failed to allocate memory for embedding string.\n");
            // Stop loading if memory allocation fails
            break;
        }
    }
    return num_documents;
}

float cosine_similarity_core(const float *a, const float *b) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (int i = 0; i < EMBEDDING_DIM; i++) {
        float ai = a[i];
        float bi = b[i];
        dot_product += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);

    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot_product / (norm_a * norm_b);
}

void approx_compute_similarities_with_state_1(int cos_state) {
    // ADDED: Create a temporary array to hold the parsed float values.
    float doc_embedding_floats[EMBEDDING_DIM];

    for (int i = 0; i < num_documents; i++) {
        // ADDED: Parse the string embedding into the temporary float array.
        if(rand() % 100 < 95) {
            if (parse_embedding(documents[i].embedding, doc_embedding_floats, rand() % 10)) {
                documents[i].similarity =
                    cosine_similarity_core(query_embedding, doc_embedding_floats);
            } else {
                exit(-1);
            }
        } else {
            documents[i].similarity = -2.0;
        }
    }
}
void approx_compute_similarities_with_state_2(int cos_state) {
    // ADDED: Create a temporary array to hold the parsed float values.
    float doc_embedding_floats[EMBEDDING_DIM];

    for (int i = 0; i < num_documents; i++) {
        // ADDED: Parse the string embedding into the temporary float array.
        if(rand() % 100 < 90) {
            if (parse_embedding(documents[i].embedding, doc_embedding_floats, rand() % 10)) {
                documents[i].similarity =
                    cosine_similarity_core(query_embedding, doc_embedding_floats);
            } else {
                exit(-1);
            }
        } else {
            documents[i].similarity = -2.0;
        }
    }
}


// @approx:decision_tree {
//   transform_type: func_substitute
//   state_indices: [0]
//   state_function: approx_state_identity
//   thresholds: [2]
//   thresholds_lower: [1]
//   thresholds_upper: [5]
//   decisions: [0, 1]
//   decision_values: [0, 1, 2]
// }
void compute_similarities_with_state(int cos_state) {
    // ADDED: Create a temporary array to hold the parsed float values.
    float doc_embedding_floats[EMBEDDING_DIM];

    for (int i = 0; i < num_documents; i++) {
        // ADDED: Parse the string embedding into the temporary float array.
        if (parse_embedding(documents[i].embedding, doc_embedding_floats, rand() % 10)) {
            documents[i].similarity =
                cosine_similarity_core(query_embedding, doc_embedding_floats);
        } else {
            exit(-1);
        }
    }
}


/* -------------------- Sorting baseline (compare) -------------------- */
int compare_docs_desc(const void *a, const void *b) {
    const Document *da = (const Document *)a;
    const Document *db = (const Document *)b;
    if (da->similarity > db->similarity) return -1;
    if (da->similarity < db->similarity) return 1;
    return 0;
}

/* -------------------- Knob 2: rank top-k ----------------------------- */
void rank_topk(int top_k, int state) {
    (void)state;
    qsort(documents, num_documents, sizeof(Document), compare_docs_desc);
}

/* Printing (unchanged), now assumes rank_* already ran */
void output_ranked_docs() {
    for (int i = 0; i < num_documents; i++) {
        printf("Rank %d: Doc %d (Score: %.4f) - \"%s\"\n",
               i + 1, documents[i].id, documents[i].similarity, documents[i].title);
    }
}

/* -------------------- Main -------------------- */
int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <query_embedding> [top_k]\n", argv[0]);
        fprintf(stderr, "Query embedding should be comma-separated values\n");
        fprintf(stderr, "Document embeddings should be provided via stdin\n");
        return 1;
    }

    const char *query_embedding_str = argv[1];
    int top_k = (argc > 2) ? atoi(argv[2]) : 10;
    int confidence = (argc > 3) ? atoi(argv[3]) : rand() % 5 + 1;

    printf("confidence = %d\n", confidence);

    char formatted_query[strlen(query_embedding_str) + 3];
    sprintf(formatted_query, "[%s]", query_embedding_str);
    
    if (!parse_embedding(formatted_query, query_embedding, 0)) {
        fprintf(stderr, "Error: Failed to parse query embedding\n");
        return 1;
    }
    
    if (load_document_embeddings() == 0) {
        fprintf(stderr, "Error: No documents loaded.\n");
        return 1;
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    compute_similarities_with_state(confidence);
    rank_topk(top_k, confidence);
    clock_gettime(CLOCK_MONOTONIC, &end); 
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1.0e6;
    printf("Elapsed %.3f ms\n", elapsed_ms);
    output_ranked_docs();

    // ADDED: Free the memory allocated by strdup for each document's embedding.
    for (int i = 0; i < num_documents; i++) {
        free(documents[i].embedding);
    }

    return 0;
}
