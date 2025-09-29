import os
import re
from typing import Callable, List, Optional, Tuple, Set
import csv
import time

# Import and initialize the ApproxMLIR SDK
from approxMLIR import ApproxMLIRSDK
sdk: ApproxMLIRSDK = ApproxMLIRSDK('./bin', './mlir', '')

# Set JAX to use GPU memory if available.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import jax
from gemma import gm

print("JAX is using device:", jax.devices()[0])

class Question:
    """Describes a question and its specific function for computing accuracy."""
    def __init__(self, text: str, accuracy_fn: Callable[[str], float]):
        self.text = text
        self.accuracy_fn = accuracy_fn

# --- Accuracy Calculation Functions ---

def acc_sum_puzzle(answer: str) -> float:
    """Accuracy for the math puzzle. The expected answer is 22."""
    if re.search(r'\b22\b', answer):
        return 1.0
    return 0.0

def acc_siebel_college(answer: str) -> float:
    """Accuracy for the Thomas Siebel question."""
    lower_answer = answer.lower()
    if ('university of illinois' in lower_answer or 
        'urbana-champaign' in lower_answer or 
        'uiuc' in lower_answer):
        return 1.0
    return 0.0

# --- Questions to run ---
questions_to_run = [
    Question(
        text="What's the sum of 2 + 5 + 10 + 3 + number of letter 'g' in the word 'piggy'.",
        accuracy_fn=acc_sum_puzzle
    ),
    Question(
        text="Where did Thomas Siebel go to college?",
        accuracy_fn=acc_siebel_college
    )
]

# --- LLM and Agent Implementation ---

class LLMManager:
    """Manages the Gemma models (fast and pro) and the sampler."""
    def __init__(self, model_sizes: List[str] = ["270m", "1b"]):
        self.models = []
        self.params = []
        self.samplers = []
        self.model_map = {size: i for i, size in enumerate(model_sizes)}
        self._load_models(model_sizes)

    def _load_models(self, model_sizes: List[str]):
        """Loads the specified Gemma models and parameters."""
        try:
            for size in model_sizes:
                if size == "270m":
                    print("Loading Gemma 270M model (fast agent)...")
                    model = gm.nn.Gemma3_270M()
                    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_270M_IT)
                elif size == "1b":
                    print("Loading Gemma 1B model (pro agent)...")
                    model = gm.nn.Gemma3_1B()
                    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_1B_IT)
                else: raise ValueError(f"Unsupported model size: {size}")
                self.models.append(model)
                self.params.append(params)
                self.samplers.append(gm.text.ChatSampler(model=model, params=params))
            print("All models loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(self, prompt: str, model_size: str) -> str:
        """Generates a response from a specific LLM."""
        model_idx = self.model_map.get(model_size)
        if model_idx is None: raise ValueError(f"Model size '{model_size}' not loaded.")
        reply = self.samplers[model_idx].chat(prompt, multi_turn=False, print_stream=False)
        return reply

def parse_qa_output(output: str) -> Tuple[str, int]:
    """Parses 'answer = ...' and 'confidence = ...' from LLM output."""
    answer_match = re.search(r"answer\s*=\s*(.*)", output, re.IGNORECASE)
    conf_match = re.search(r"confidence\s*=\s*(\d+)", output, re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else "Parsing failed."
    confidence = int(conf_match.group(1)) if conf_match else 0
    return answer, confidence

def parse_reflection_output(output: str) -> Tuple[str, int]:
    """Parses 'final_answer = ...' and 'new_confidence = ...' from reflection output."""
    answer_match = re.search(r"final_answer\s*=\s*(.*)", output, re.IGNORECASE)
    conf_match = re.search(r"new_confidence\s*=\s*(\d+)", output, re.IGNORECASE)
    final_answer = answer_match.group(1).strip() if answer_match else "Reflection parsing failed."
    new_confidence = int(conf_match.group(1)) if conf_match else 0
    return final_answer, new_confidence

def organize_and_run_adaptive_agent(question: Question, llm: LLMManager, max_loops: int = 3) -> Tuple[float, float]:
    """Orchestrates the adaptive, reflection-based agent workflow using a complete SDK-controlled loop."""
    total_time = 0
    current_confidence_state = 3  # Initial state for the first iteration
    final_answer = "Agent failed to produce a final answer."
    loop_count = 0

    while loop_count < max_loops:
        loop_count += 1
        print(f"\n--- Loop Iteration {loop_count} ---")

        # 1. Knob 1: Select QA agent based on current confidence state
        agent_choice = sdk.get_knob_val(1, current_confidence_state)
        agent_name = "Pro Agent"
        selected_agent_size = "1b"
        if agent_choice == 1:
            selected_agent_size = "270m"
            agent_name = "Fast Agent"
        print(f"Knob '1' with input '{current_confidence_state}' selected: {agent_name}")

        # 2. Run selected QA Agent
        qa_prompt = f"Answer the following question and provide a confidence score (1-5).\n\nFormat:\nanswer = <Your answer>\nconfidence = <Your score>\n\nQuestion: {question.text}"
        start_time = time.time()
        response = llm.generate(qa_prompt, model_size=selected_agent_size)
        total_time += time.time() - start_time
        answer, confidence = parse_qa_output(response)
        print(f"{agent_name} Answer: '{answer}' (Confidence: {confidence})")

        # 3. Knob 2: Decide whether to exit or reflect
        exit_choice = sdk.get_knob_val(2, confidence)
        if exit_choice == 1:
            print(f"\nKnob '2' with input '{confidence}' decided to EXIT. Finalizing answer.")
            final_answer = answer
            break
        else:
            print(f"\nKnob '2' with input '{confidence}' decided to REFLECT.")
            reflection_prompt = f"An agent answered the following question with confidence {confidence}:\nQuestion: {question.text}\nAnswer: {answer}\n\nAnalyze and provide a new, refined answer and a new confidence score.\n\nFormat:\nreflection = <Your analysis>\nfinal_answer = <Your new answer>\nnew_confidence = <Your new score>"
            
            start_time = time.time()
            reflection_response = llm.generate(reflection_prompt, model_size="1b") # Reflection uses pro model
            total_time += time.time() - start_time
            
            reflected_answer, new_confidence = parse_reflection_output(reflection_response)
            
            # Update state for the next loop iteration
            current_confidence_state = new_confidence
            final_answer = reflected_answer # Keep the most recent answer
            print(f"Reflection complete. New confidence state for next loop: {current_confidence_state}")

    if loop_count >= max_loops:
        print("\nMax loop iterations reached. Exiting with the last known answer.")

    # 4. Compute Accuracy
    accuracy = question.accuracy_fn(final_answer)
    print(f"\nFinal Answer Used for Grading: {final_answer}")
    print(f"Computed Accuracy: {accuracy:.2f}")
    print(f"Total Generation Time: {total_time:.2f}s")
    
    return accuracy, total_time

def get_acc_perf(questions):
    """Runs all questions and computes average accuracy and performance."""
    acc_sum, perf_sum = 0, 0
    llm_manager = LLMManager(model_sizes=["270m", "1b"])
    for i, question in enumerate(questions):
        print(f"\n\n===== Processing Question {i+1}: '{question.text}' =====")
        acc, perf = organize_and_run_adaptive_agent(question, llm_manager)
        acc_sum += acc
        perf_sum += perf
    avg_acc = acc_sum / len(questions) if questions else 0
    avg_perf = perf_sum / len(questions) if questions else 0
    return avg_acc, avg_perf

if __name__ == "__main__":
    avg_accuracy, avg_performance = get_acc_perf(questions_to_run)
    print("\n\n===== FINAL RESULTS =====")
    print(f"Average Accuracy: {avg_accuracy:.2f}")
    print(f"Average Performance (Time): {avg_performance:.2f}s")