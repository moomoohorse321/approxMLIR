import os
import subprocess
import re
from typing import Callable, List, Optional, Any, Tuple, Set
import csv
import time
from approxMLIR import ToolBox

sdk = ApproxMLIRSDK('./bin', './mlir')

# Set JAX to use GPU memory if available.
# This is based on the provided notebook.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import jax
from gemma import gm

print("JAX is using device:", jax.devices()[0])

truncate_fn = lambda out: out.strip()[:4096]

def strip_quotes_from_args(args: List[str]) -> List[str]:
    """Removes leading/trailing single and double quotes from each argument."""
    return [arg.strip().strip("'\"") for arg in args]

class Tool:
    """
    Describes a tool: its name, what it does, how to execute it,
    and an optional function to process its output. [cite: 389]
    """
    def __init__(self, name: str, description: str, path: str, 
                 post_processing_fn: Optional[Callable[[str], str]] = None,
                 command_parsing_fn: Optional[Callable[[List[str]], List[str]]] = None):
        self.name = name
        self.description = description
        self.path = path
        self.post_processing_fn = post_processing_fn
        self.command_parsing_fn = command_parsing_fn 
        

class Question:
    """
    Describes a question and its specific function for computing accuracy. [cite: 390]
    """
    def __init__(self, text: str, accuracy_fn: Callable[[str], float]):
        self.text = text
        self.accuracy_fn = accuracy_fn

# Define the available tools
available_tools = [
    Tool(
        name="blackscholes",
        description="Calculate the price of options. Usage: blackscholes(1, <input_path>, <output_path>)",
        path="bin/blackscholes.exec",
        post_processing_fn=truncate_fn,
        command_parsing_fn=strip_quotes_from_args
    ), 
    Tool(
        name="kmeans",
        description="Find K centriods of N nodes (assuming tool already knows input and output). Usage kmeans(-k, <number of clusers>, -n, <number of nodes>)",
        path="bin/kmeans.exec",
        post_processing_fn=truncate_fn,
        command_parsing_fn=strip_quotes_from_args
    ),
    Tool(
        name="lavaMD",
        description="find molecule movement of N random particles (assuming tool already knows input and output). Usage lavaMD(-boxes1d, <number of particles>)",
        path="bin/lavaMD.exec",
        post_processing_fn=truncate_fn,
        command_parsing_fn=strip_quotes_from_args
    ),
    Tool(
        name="pagerank",
        description="rank the web-pages of N websites (assuming tool already knows input and output). Usage pagerank(-p, -n, <number of pages>)",
        path="bin/pagerank.exec",
        post_processing_fn=truncate_fn,
        command_parsing_fn=strip_quotes_from_args
    )
]

################## start grading #########################

def _find_best_match_float(text: str, expected_val: float) -> float | None:
    """
    Finds all numbers in the text and returns the one closest to the expected value.
    This makes the parsing tolerant to conversational text from the LLM.
    """
    # This regex correctly identifies integers and floating-point numbers.
    found_numbers = [float(num) for num in re.findall(r'-?\d+(?:\.\d+)?', text)]
    if not found_numbers:
        return None

    # Find the number in the list that is closest to the expected value
    best_match = min(found_numbers, key=lambda val: abs(val - expected_val))
    return best_match

def _calculate_graded_accuracy(answer: str, expected_val: float, worst_val: float) -> float:
    """
    Calculates a graded accuracy score using the best matching number found in the text.
    """
    actual_val = _find_best_match_float(answer, expected_val)
    if actual_val is None:
        return 0.0

    error = abs(actual_val - expected_val)
    max_error = abs(worst_val - expected_val)

    if max_error == 0:
        return 1.0 if error < 1e-9 else 0.0

    normalized_error = min(1.0, error / max_error)
    return 1.0 - normalized_error

# --- Specific Accuracy Functions ---

def acc_blackscholes_avg(answer: str) -> float:
    """Accuracy for average Black-Scholes price."""
    expected_avg = 20.804847
    worst_avg = 182.633818 # The max price is furthest from the average
    return _calculate_graded_accuracy(answer, expected_avg, worst_avg)

def acc_blackscholes_max_10(answer: str) -> float:
    """Accuracy for max price of the first 10 options."""
    expected_max = 102.641959
    worst_val = 0.008416 # Min price of the first 10
    return _calculate_graded_accuracy(answer, expected_max, worst_val)

def acc_kmeans_centroids(answer: str) -> float:
    """Accuracy is the percentage of correctly identified centroids."""
    ground_truth_centroids: Set[Tuple[float, float]] = {
        (82.54, 84.45), (47.23, 15.19), (45.17, 56.70), (15.88, 15.40),
        (15.83, 82.18), (63.97, 36.69), (84.65, 15.71), (15.91, 46.67),
        (49.36, 85.72), (85.45, 53.82)
    }
    found_centroids = re.findall(r"\(([\d.]+),\s*([\d.]+)\)", answer)
    parsed_set = {(float(x), float(y)) for x, y in found_centroids}

    matches = 0
    for gt_c in ground_truth_centroids:
        if any(abs(p_c[0] - gt_c[0]) < 0.1 and abs(p_c[1] - gt_c[1]) < 0.1 for p_c in parsed_set):
            matches += 1
    return matches / 10.0

def acc_kmeans_leftmost(answer: str) -> float:
    """Graded accuracy for the leftmost KMeans centroid by finding the best matching x-coordinate."""
    expected_x = 15.83  # Leftmost point
    worst_x = 85.45      # Rightmost point

    # Find all tuples and extract their x-coordinates
    found_tuples = re.findall(r"\(([\d.]+),\s*[\d.]+\)", answer)
    if not found_tuples:
        return 0.0

    # Find the x-coordinate closest to the expected one
    actual_x = min([float(x) for x in found_tuples], key=lambda val: abs(val - expected_x))

    error = abs(actual_x - expected_x)
    max_error = abs(worst_x - expected_x)

    normalized_error = min(1.0, error / max_error)
    return 1.0 - normalized_error

def acc_lavamd_avg_energy(answer: str) -> float:
    """Graded accuracy for the average potential energy."""
    expected_energy = 1322.079678
    worst_energy = 294.916728 # Min energy is furthest from the average
    return _calculate_graded_accuracy(answer, expected_energy, worst_energy)

def acc_pagerank_max(answer: str) -> float:
    """Graded accuracy for the max PageRank score."""
    expected_rank = 0.000004325491 # Max Rank
    worst_rank = 0.000001404082   # Min Rank
    return _calculate_graded_accuracy(answer, expected_rank, worst_rank)

def acc_pagerank_top_10(answer: str) -> float:
    """Accuracy is the percentage of correctly identified top 10 nodes."""
    top_10_nodes = {
        "122588", "618652", "187351", "442834", "104308",
        "546645", "578235", "580821", "454897", "139527"
    }
    found_numbers = set(re.findall(r'\b\d{5,}\b', answer)) # Find standalone numbers with 5+ digits
    matches = len(top_10_nodes.intersection(found_numbers))
    return matches / 10.0

def acc_sum_puzzle(answer: str) -> float:
    """
    Accuracy for the math puzzle. The expected answer is 22.
    (2 + 5 + 10 + 3 + 2 'g's in 'piggy' = 22)
    """
    # Use regex to find the whole number '22' to avoid matching '122', etc.
    if re.search(r'\b22\b', answer):
        return 1.0
    return 0.0

def acc_siebel_college(answer: str) -> float:
    """
    Accuracy for the Thomas Siebel question. He attended the University of Illinois.
    Checks for variations like 'University of Illinois', 'Urbana-Champaign', or 'UIUC'.
    """
    lower_answer = answer.lower()
    if ('university of illinois' in lower_answer or 
        'urbana-champaign' in lower_answer or 
        'uiuc' in lower_answer):
        return 1.0
    return 0.0

################## end grading #########################


questions_to_run = [
    # --- Blackscholes Questions ---
    Question(
        text="Calculate the prices for the options listed in 'data/input.bin' and write the results to 'data/out.txt'. What is the average prices of all the options?",
        accuracy_fn=acc_blackscholes_avg
    ),
    Question(
        text="I need to price a batch of 10000000 financial options from 'data/input.bin' and save them. What is the maximum price of the first 10 options?",
        accuracy_fn=acc_blackscholes_max_10
    ),
    # --- KMeans Questions ---
    Question(
        text="I have 1000000 data points that I need to group into 10 distinct clusters. Can you compute the 10 centroids?",
        accuracy_fn=acc_kmeans_centroids
    ),
    Question(
        text="What is the left most centroid among the 10 centroids of the input file. The input file has 1000000 nodes.",
        accuracy_fn=acc_kmeans_leftmost
    ),
    # --- LavaMD Questions ---
    Question(
        text="Run a molecular dynamics simulation for a system containing 10 particles dictated by input files, what's their average energy.",
        accuracy_fn=acc_lavamd_avg_energy
    ),
    Question(
        text="How would I simulate the molecular movement of 12 random particles using the available tools?",
        accuracy_fn=lambda a: 1.0 if 'lavaMD(-boxes1d,12)' in a.replace(" ", "") else 0.0
    ),
    # --- PageRank Questions ---
    Question(
        text="I need to calculate the PageRank for a network of 500000 web pages. What's the largest rank?",
        accuracy_fn=acc_pagerank_max
    ),
    Question(
        text="Rank 1000000 websites based on their importance in a network, what are the top 10 pages?",
        accuracy_fn=acc_pagerank_top_10
    ),
    # -- Non-tool questions ---- 
    Question(
        text="What's the sum of 2 + 5 + 10 + 3 + number of letter 'g' in the word 'piggy'.",
        accuracy_fn=acc_sum_puzzle
    ),
    Question(
        text="Where did Thomas Siebel go to college?",
        accuracy_fn=acc_siebel_college
    )
]

class LLMManager:
    """
    A class to manage the Gemma model and sampler, ensuring it's loaded only once.
    This encapsulates the model inference logic as requested, so the backend
    can be easily swapped with a compiled IREE artifact.
    """
    def __init__(self, model_size: List = ["270m"]):
        self.model = []
        self.params = []
        self.sampler = []
        self.model_used = 1
        self.model_size = model_size
        self._load_model()

    def _load_model(self):
        """Loads the Gemma model and parameters based on the specified size."""
        try:
            if "270m" in self.model_size:
                print("Loading Gemma 270M model architecture...")
                self.model.append(gm.nn.Gemma3_270M())
                print("Loading 270M model parameters...")
                self.params.append(gm.ckpts.load_params(
                    gm.ckpts.CheckpointPath.GEMMA3_270M_IT
                ))
            if "1b" in self.model_size:
                print("Loading Gemma 1B model architecture...")
                self.model.append(gm.nn.Gemma3_1B())
                print("Loading 1B model parameters...")
                self.params.append(gm.ckpts.load_params(
                    gm.ckpts.CheckpointPath.GEMMA3_1B_IT
                ))
            if "4b" in self.model_size:
                # As a fallback or for other models like 4B from the notebook
                print(f"Loading Gemma {self.model_size} model architecture...")
                self.model = gm.nn.Gemma3_4B()
                self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

            print("Model and parameters loaded successfully.")

            # The ChatSampler is the easiest way to prompt the model, handling
            # conversation formatting automatically.
            for i in range(0, len(self.model)):
                self.sampler.append(gm.text.ChatSampler(
                    model=self.model[i],
                    params=self.params[i],
                ))
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have authenticated with Kaggle and have the necessary permissions.")
            raise

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the LLM given a prompt.
        This is the core function to be replaced by your compiled artifact's inference call.
        """
        
        # Using multi_turn=False to ensure each generation is independent
        reply = self.sampler[self.model_used].chat(prompt, multi_turn=False, print_stream=False)
        return reply




def tool_invocation(llm_output: str, tools: List[Tool]) -> str:
    """
    Parses the LLM output to find a tool command, invokes the tool,
    and returns its post-processed output. [cite: 393]
    """
    # Simple regex to find a command like: tool_name(arg1, "arg 2", ...)
    match = re.search(r'(\w+)\((.*)\)', llm_output)

    if not match:
        return "No tool was invoked. Could not parse command."

    tool_name, args_str = match.groups()
    
    if args_str.strip():
        # csv.reader expects an iterable, so we pass the string as a single-element list.
        # skipinitialspace=True handles cases like '5, 7' instead of '5,7'.
        args = next(csv.reader([args_str], skipinitialspace=True))
    else:
        args = []

    selected_tool = next((t for t in tools if t.name == tool_name), None)

    if not selected_tool:
        return f"Error: Tool '{tool_name}' not found."
    
    if hasattr(selected_tool, "command_parsing_fn") and selected_tool.command_parsing_fn:
        args = selected_tool.command_parsing_fn(args)
    try:
        command = [selected_tool.path] + args
        print(f"Invoking tool: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        raw_output = result.stdout.strip()
        
        if selected_tool.post_processing_fn:
            return selected_tool.post_processing_fn(raw_output)
        return raw_output

    except FileNotFoundError:
        return f"Error: The executable for '{tool_name}' was not found at '{selected_tool.path}'."
    except subprocess.CalledProcessError as e:
        return f"Error executing tool '{tool_name}': {e.stderr}, {e.stdout}"
    except Exception as e:
        return f"An unexpected error occurred during tool invocation: {e}"


def organize_and_run_agent(question: Question, tools: List[Tool], llm: LLMManager) -> Tuple[str, float]:
    """
    Orchestrates the entire agent workflow. [cite: 392]
    1.  LLM selects a tool.
    2.  The tool is invoked.
    3.  LLM generates a final answer based on the tool's output.
    4.  The answer's accuracy is calculated.
    """
    generation_time = 0
    tool_time = 0
    # Step 1: Create a prompt for the LLM to choose a tool.
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    tool_selection_prompt = f"""
You have access to the following tools:
{tool_descriptions}

Your task is to answer the user's question. First, decide if a tool needs to be invoked. A tool needs to be invoked only if its output can help answer the question.
If so, respond with ONLY the command to call the tool.
Also compute a score out of 5 for necessity of using a tool.

For example if there are 3 tools:

- X: Calculator. Usage: X(1, <input_path>, <output_path>)
- Y: Info retriever. Usage Y(-k, <number of clusers>, -n, <number of nodes>)
- Z: Physics simulator. Usage Z(-boxes1d, <number of particles>)

A possible answer to "calculate x + y, x in ./x.txt y in ./y.txt" can be 

cmd = X(1, ./x.txt, ./y.txt)
necessity = 3

A possible answer to "What is UIUC?" can be 

cmd = No tool is needed.
necessity = 0

Question: {question.text}
"""
    print("\n--- Generating LLM Response ---")
    print(f"PROMPT:\n{tool_selection_prompt}")
    print("------------------------------")
    
    # Ask LLM to select a tool
    start_time = time.time()
    llm_tool_choice = llm.generate(tool_selection_prompt)
    generation_time += time.time() - start_time
    print(f"\nLLM Tool Choice: {llm_tool_choice}")
    
    # Use regex to find 'cmd = ...' line, ignoring case
    cmd_match = re.search(r"cmd\s*=\s*(.*)", llm_tool_choice, re.IGNORECASE)
    if cmd_match:
        command_str = cmd_match.group(1).strip()

    # Use regex to find 'necessity = ...' line, ignoring case
    nec_match = re.search(r"necessity\s*=\s*(\d+)", llm_tool_choice, re.IGNORECASE)
    if nec_match:
        necessity_score = int(nec_match.group(1))
        
    # call the knob
    

    # Step 2: Invoke the tool
    tool_output = ""
    # Check if the model explicitly stated no tool is needed.
    if sdk.get_knob_val(1, necessity_score):
        print("LLM decided no tool was needed.")
        # The final prompt will use this context.
        context_from_tool = ""
        usefulness_score = 0
    else:
        # If a tool was chosen, invoke it.
        start_time = time.time()
        tool_output = tool_invocation(command_str, tools)
        tool_time += time.time() - start_time
        print(f"Processed Tool Output: {tool_output}")
        # The final prompt will use the tool's output.
        context_from_tool = f'To help answer your question, you have received the following information from a tool: "{tool_output}"'

        summarize_tool_output_prompt = f"""
        Question: {question.text}

        {context_from_tool}
        
        You must summarize the output such that the info that's related to the question is kept. 
        
        You must also compute a number in fomrat of "usefulness = <how useful the tool output is>" suggesting if the tool's output is useful or not.
    """

        start_time = time.time()
        summary = llm.generate(summarize_tool_output_prompt)
        context_from_tool = f"To help answer your question, you have received the following information from a tool: {summary}"
        generation_time += time.time() - start_time
        
        useful_match = re.search(r"usefulness\s*=\s*(\d+)", llm_tool_choice, re.IGNORECASE)
        if useful_match:
            usefulness_score = int(useful_match.group(1))

    # Step 3: Generate the final answer using the tool's context
    final_answer_prompt = f"""
Original Question: {question.text}

{context_from_tool}

Based on tool's output, provide a clear and concise final answer to the original question. 

You must not output any code to answer the question. 

Assume all the input and output are analyzed by the tools invoked.

Final Answer:
"""

    print("\n--- Generating LLM Response ---")
    print(f"PROMPT:\n{final_answer_prompt}")
    print("------------------------------")
    
    start_time = time.time()
    llm.model_used = sdk.get_knob_val(2, usefulness_score) - 1
    final_answer = llm.generate(final_answer_prompt)
    generation_time += time.time() - start_time
    
    print(f"\nFinal Generated Answer: {final_answer}")

    # Step 4: Compute accuracy
    accuracy = question.accuracy_fn(final_answer)
    print(f"Computed Accuracy: {accuracy:.2f}")
    
    return accuracy, tool_time + generation_time

def get_acc_perf(questions):
    acc_sum = 0
    perf_sum = 0
    for i, question in enumerate(questions):
        print(f"\n\n===== Processing Question {i+1}: '{question.text}' =====")
        acc, perf = organize_and_run_agent(question, available_tools, llm_manager)
        acc_sum += acc
        perf_sum += perf
    return acc_sum / len(questions), perf_sum / len(questions)
    
if __name__ == "__main__":
    # here is an example on auto-tuner side (how to invoke it)
    # the dependency of this file is the *.exec in the binary directory.
    llm_manager = LLMManager(model_size=["270m", "1b"])
    
    print(get_acc_perf(questions_to_run))
    