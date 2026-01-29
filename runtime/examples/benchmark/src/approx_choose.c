
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>   // for strcasecmp
#include <ctype.h>
#include <math.h>
#include <time.h>

int approx_state_identity(int state) { return state; }

/**
 * @brief Chooses a model based on input. The logic here is a placeholder,
 * as it will be replaced by the approxMLIR task_skipping transformation.
 *
 * @param input An arbitrary input to the function.
 * @param state An integer state passed from the caller, used by approxMLIR
 * to determine which approximate version to execute. This
 * argument is mandatory for any function targeted by a knob.
 * @return An integer representing the chosen model.
 */
// @approx:decision_tree {
//   transform_type: task_skipping
//   state_indices: [2]
//   state_function: approx_state_identity
//   thresholds: [2]
//   thresholds_lower: [0]
//   thresholds_upper: [4]
//   decisions: [1, 2]
//   decision_values: [0, 1, 2]
// }
void model_choose(int input, int* output, int state) {
    // This logic is arbitrary and will be overwritten by the approximation.
    // It's here to make the original program compilable and valid.
    if (input > 10) {
        *output = 1;
    } else if(input > 20){
        *output = 2; // Represents choosing model 2
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <state>\n", argv[0]);
        return 1;
    }

    // The state is read from the command-line arguments. This state will be
    // used by the approxMLIR runtime to select a code path.
    int state = atoi(argv[1]);
    int some_input = 15; // An example input value.

    int chosen_model;
    model_choose(some_input, &chosen_model, state);

    printf("Based on the input and state, the chosen model is: %d\n", chosen_model);

    return 0;
}
