// RUN: cgeist -O0 %stdinclude %s -S > %s.mlir 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h> // For tolower
#include <time.h> 


// --- BM25 Parameters ---
const double K1 = 1.5; // Typically between 1.2 and 2.0
const double B = 0.75; // Typically around 0.75

// --- Helper Struct for Ranking ---
typedef struct {
    int doc_index;
    double score;
} DocumentScore;

// --- Helper Function: Case-insensitive string comparison ---
// Simple token comparison (case-insensitive)
int compare_tokens(const char *t1, const char *t2) {
    return strcasecmp(t1, t2); // strcasecmp is POSIX; use _stricmp on Windows
}

// --- Simple Tokenizer & Word Counter ---
// Counts words (tokens) in a string based on spaces and basic punctuation.
// Converts the string to lowercase in the process!
int count_and_lower_words(char *str) {
    if (str == NULL || *str == '\0') return 0;
    int count = 0;
    char *p = str;
    int in_word = 0;

    while (*p) {
        *p = tolower((unsigned char)*p); // Convert to lowercase
        if (isalnum((unsigned char)*p)) { // Part of a word
            if (!in_word) {
                in_word = 1;
                count++;
            }
        } else { // Not part of a word (space, punctuation)
            in_word = 0;
        }
        p++;
    }
    return count;
}

// --- Term Frequency (TF) Calculation ---
// Counts occurrences of a term (case-insensitive) in a document string.
// Assumes the document string has already been lowercased.
int calculate_tf(const char *term, const char *doc_lower) {
    if (term == NULL || doc_lower == NULL) return 0;

    int count = 0;
    const char *p = doc_lower;
    size_t term_len = strlen(term);

    while ((p = strstr(p, term)) != NULL) {
        // Check for whole word match (surrounded by non-alphanumeric or start/end)
        int is_start = (p == doc_lower || !isalnum((unsigned char)*(p - 1)));
        int is_end = (*(p + term_len) == '\0' || !isalnum((unsigned char)*(p + term_len)));

        if (is_start && is_end) {
            count++;
        }
        p += term_len; // Move past the found term
    }
    return count;
}

// --- Document Frequency (DF) Calculation ---
// Counts how many documents in the corpus contain the term (case-insensitive).
// Modifies document copies by converting to lowercase.
int calculate_df(const char *term, char **corpus, int num_docs) {
    if (term == NULL || corpus == NULL) return 0;

    int count = 0;
    char *lower_term = strdup(term); // Work with lowercase term
    if (!lower_term) return 0; // Allocation failed
    for (char *lt = lower_term; *lt; ++lt) *lt = tolower((unsigned char)*lt);

    for (int i = 0; i < num_docs; ++i) {
        char *doc_copy = strdup(corpus[i]); // Create a modifiable copy
        if (!doc_copy) continue; // Allocation failed, skip doc

        // Convert doc copy to lowercase
        for (char *p = doc_copy; *p; ++p) *p = tolower((unsigned char)*p);

        const char *p = doc_copy;
        size_t term_len = strlen(lower_term);
        int found_in_doc = 0;

         while ((p = strstr(p, lower_term)) != NULL) {
             int is_start = (p == doc_copy || !isalnum((unsigned char)*(p - 1)));
             int is_end = (*(p + term_len) == '\0' || !isalnum((unsigned char)*(p + term_len)));

             if (is_start && is_end) {
                 found_in_doc = 1;
                 break; // Found it once, no need to search more in this doc
             }
             p += term_len;
         }

        if (found_in_doc) {
            count++;
        }
        free(doc_copy);
    }
    free(lower_term);
    return count;
}

// --- Inverse Document Frequency (IDF) Calculation ---
// Using a common BM25 IDF variant to avoid log(0) or division by zero.
double calculate_idf(int df, int num_docs) {
    // Ensure N is at least df
    double N = (double)num_docs;
    double n_q = (double)df;
    // Formula: log( (N - n_q + 0.5) / (n_q + 0.5) + 1 ) -- adding 1 avoids issues if n_q = N
    // Note: +1 inside log is common to ensure positive IDF, other variants exist.
    double idf = log( ( (N - n_q + 0.5) / (n_q + 0.5) ) + 1.0 );
    return (idf > 0.0) ? idf : 0.0; // Return 0 for terms not in corpus or negative IDF (shouldn't happen with +1)
}

// --- Comparison function for qsort ---
int compare_scores(const void *a, const void *b) {
    DocumentScore *scoreA = (DocumentScore *)a;
    DocumentScore *scoreB = (DocumentScore *)b;
    // Sort descending (higher score first)
    if (scoreB->score > scoreA->score) return 1;
    if (scoreB->score < scoreA->score) return -1;
    return 0;
}

// --- Main BM25 Ranking Function ---
DocumentScore* rank_documents_bm25(char *query, char **corpus, int num_docs) {
    if (query == NULL || corpus == NULL || num_docs <= 0) {
        return NULL;
    }

    // 1. Preprocessing: Calculate document lengths and average length
    double *doc_lengths = (double *)malloc(num_docs * sizeof(double));
    char **lower_corpus = (char **)malloc(num_docs * sizeof(char*)); // Store lowercase docs
    if (!doc_lengths || !lower_corpus) {
        fprintf(stderr, "Error allocating memory for preprocessing.\n");
        free(doc_lengths); free(lower_corpus); // Free potentially allocated parts
        return NULL;
    }

    double total_doc_length = 0;
    for (int i = 0; i < num_docs; ++i) {
        char *doc_copy = strdup(corpus[i]);
        if (!doc_copy) {
             fprintf(stderr, "Error duplicating document %d.\n", i);
             // Clean up previously allocated copies
             for(int j = 0; j < i; ++j) free(lower_corpus[j]);
             free(lower_corpus); free(doc_lengths);
             return NULL;
        }
        doc_lengths[i] = (double)count_and_lower_words(doc_copy); // Modifies doc_copy
        total_doc_length += doc_lengths[i];
        lower_corpus[i] = doc_copy; // Store the lowercased copy
    }
    double avg_doc_len = total_doc_length / num_docs;

    // 2. Allocate space for scores
    DocumentScore *scores = (DocumentScore *)malloc(num_docs * sizeof(DocumentScore));
    if (!scores) {
        fprintf(stderr, "Error allocating memory for scores.\n");
        for(int i = 0; i < num_docs; ++i) free(lower_corpus[i]);
        free(lower_corpus); free(doc_lengths);
        return NULL;
    }

    // Initialize scores
    for (int i = 0; i < num_docs; ++i) {
        scores[i].doc_index = i;
        scores[i].score = 0.0;
    }

    // 3. Process Query: Tokenize and calculate score for each term
    char *query_copy = strdup(query);
    if (!query_copy) {
         fprintf(stderr, "Error duplicating query.\n");
         for(int i = 0; i < num_docs; ++i) free(lower_corpus[i]);
         free(lower_corpus); free(doc_lengths); free(scores);
         return NULL;
    }
    // We don't need to lowercase query_copy here, tf/df handle case internally

    char *term;
    char *rest = query_copy;
    char delimiters[] = " .,;:!?\"\'\n\t()[]{}<>"; // Punctuation and whitespace

    // Simple way to handle unique terms: keep track of processed terms
    char *processed_terms[100]; // Max 100 unique terms - adjust if needed
    int processed_count = 0;

    while (1) {
        if(!(term = strtok_r(rest, (char*)delimiters, &rest))) {
            break; // No more tokens
        }
        if (strlen(term) == 0) continue; // Skip empty tokens

        // Check if term was already processed (case-insensitive)
        int already_processed = 0;
        for (int k = 0; k < processed_count; ++k) {
            if (compare_tokens(term, processed_terms[k]) == 0) {
                already_processed = 1;
                break;
            }
        }
        if (already_processed) continue;

        // Store the processed term (allocating memory)
        if (processed_count < 100) {
             processed_terms[processed_count] = strdup(term);
             if (processed_terms[processed_count]) { // Check allocation
                 processed_count++;
             } else {
                 fprintf(stderr, "Warning: Could not allocate memory for processed term '%s'. Skipping.\n", term);
             }
        } else {
            fprintf(stderr, "Warning: Exceeded maximum unique query terms (100).\n");
            break; // Stop processing more terms
        }


        // Calculate DF and IDF for the term (DF is expensive here)
        // Note: calculate_df takes original corpus, not lower_corpus
        int df = calculate_df(term, corpus, num_docs);
        if (df == 0) continue; // Term not in any document, skip

        double idf = calculate_idf(df, num_docs);

        // Calculate term's contribution to each document's score
        for (int i = 0; i < num_docs; ++i) {
            // Calculate TF using the lowercased term and lowercased document
            char *lower_term = strdup(term);
            if (!lower_term) continue; // Allocation failed
            for (char *lt = lower_term; *lt; ++lt) *lt = tolower((unsigned char)*lt);

            int tf = calculate_tf(lower_term, lower_corpus[i]);
            free(lower_term); // Free the lowercase term copy

            // BM25 term score calculation
            double numerator = (double)tf * (K1 + 1.0);
            double denominator = (double)tf + K1 * (1.0 - B + B * (doc_lengths[i] / avg_doc_len));
            double term_score = idf * (numerator / denominator);

            scores[i].score += term_score;
        }
    }

    // 4. Clean up temporary allocations
    free(query_copy);
    for (int i = 0; i < num_docs; ++i) {
        free(lower_corpus[i]); // Free the lowercase doc copies
    }
    free(lower_corpus);
    free(doc_lengths);
     for (int k = 0; k < processed_count; ++k) {
        free(processed_terms[k]); // Free the processed term copies
    }


    // // 5. Sort documents by score (descending)
    // qsort(scores, num_docs, sizeof(DocumentScore), compare_scores);

    return scores;
}

char **generate_corpus(int num_docs, unsigned int seed, int min_words, int max_words) {
    if (num_docs <= 0 || min_words <= 0 || max_words < min_words) return NULL;

    static const char *vocab[] = {
        "the","quick","brown","fox","jumps","over","lazy","dog",
        "search","engine","retrieval","ranking","bm25","information",
        "system","text","document","query","model","term","frequency",
        "inverse","document","probabilistic","score","token","index",
        "vector","space","tfidf","normalize","length","field","weight",
        "analysis","embedding","graph","random","walk","pagerank","quality",
        "relevance","language","modeling","approximate","distance","measure"
    };
    const int V = (int)(sizeof(vocab)/sizeof(vocab[0]));

    srand(seed);

    char **corpus = (char**)malloc((size_t)num_docs * sizeof(char*));
    if (!corpus) return NULL;

    for (int i = 0; i < num_docs; ++i) {
        int words = min_words + (rand() % (max_words - min_words + 1));

        /* conservative buffer: (max 10 chars per word + 1 space) * words + period + NUL */
        size_t cap = (size_t)words * 12 + 2;
        char *buf = (char*)malloc(cap);
        if (!buf) {
            for (int j = 0; j < i; ++j) free(corpus[j]);
            free(corpus);
            return NULL;
        }

        size_t len = 0;
        for (int w = 0; w < words; ++w) {
            const char *word = vocab[rand() % V];
            int wrote = 0;
            if (w == 0) {
                wrote = snprintf(buf + len, cap - len, "%s", word);
            } else {
                wrote = snprintf(buf + len, cap - len, " %s", word);
            }
            if (wrote < 0) { wrote = 0; } /* guard */
            len += (size_t)wrote;
            if (len >= cap) { len = cap - 1; break; }
        }

        /* add trailing period if space remains */
        if (len + 1 < cap) {
            buf[len++] = '.';
            buf[len] = '\0';
        } else {
            buf[cap - 1] = '\0';
        }

        corpus[i] = buf;
    }

    return corpus;
}

void free_corpus(char **corpus, int num_docs) {
    if (!corpus) return;
    for (int i = 0; i < num_docs; ++i) free(corpus[i]);
    free(corpus);
}

// --- Main Function (Example Usage) ---

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s <num_docs> [seed] [\"query string\"]\n", prog);
    fprintf(stderr, "  num_docs     : number of documents to generate (default 6)\n");
    fprintf(stderr, "  seed         : RNG seed (default 42)\n");
    fprintf(stderr, "  query string : BM25 query (default \"quick brown fox\")\n");
}

/// ./bm25 6 42 "quick brown fox" (<num doc> <seed> <query>)
int main(int argc, char **argv) {
    int num_docs = 6;
    unsigned int seed = 42;
    char *query; // = "quick brown fox";

    if (argc >= 2) {
        num_docs = atoi(argv[1]);
        if (num_docs <= 0) {
            fprintf(stderr, "num_docs must be positive.\n");
            usage(argv[0]);
            return 1;
        }
    }
    if (argc >= 3) {
        seed = (unsigned int)strtoul(argv[2], NULL, 10);
    }
    if (argc >= 4) {
        query = argv[3];
    }

    // int num_docs = sizeof(corpus) / sizeof(corpus[0]);

    printf("Query: \"%s\"\n\n", query);
    printf("Ranking documents:\n");

    char **corpus = generate_corpus(num_docs, seed, 6, 16);
    if (!corpus) { fprintf(stderr, "generate_corpus failed\n"); return 1; }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    DocumentScore *ranked_scores = rank_documents_bm25(query, corpus, num_docs);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1.0e6;

    if (ranked_scores) {
        for (int i = 0; i < num_docs; ++i) {
            int doc_index = ranked_scores[i].doc_index;
            printf("Rank %d: Doc %d (Score: %.4f) - \"%s\"\n",
                   i + 1,
                   doc_index,
                   ranked_scores[i].score,
                   corpus[doc_index]); // Print original document
        }
        free(ranked_scores); // Free the results array
    } else {
        printf("An error occurred during ranking.\n");
    }

    printf("\nComputation time: %.3f ms\n", elapsed_ms);

    return 0;
}

// /home/hao/Polygeist/build/bin/cgeist -resource-dir=/home/hao/Polygeist/llvm-project/build/lib/clang/18 -I /home/hao/Polygeist/tools/cgeist/Test/polybench/utilities /home/hao/Polygeist/tools/cgeist/Test/approxMLIR/bm25.c -S --function=rank_documents_bm25