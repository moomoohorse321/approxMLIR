import os
import subprocess
import re
from typing import Callable, List, Optional, Any, Tuple
import csv

# Set JAX to use GPU memory if available.
# This is based on the provided notebook.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import jax
from gemma import gm

print("JAX is using device:", jax.devices()[0])

truncate_fn = lambda out: out.strip()[:4096]

class Tool:
    """
    Describes a tool: its name, what it does, how to execute it,
    [cite_start]and an optional function to process its output. [cite: 389]
    """
    def __init__(self, name: str, description: str, path: str, post_processing_fn: Optional[Callable[[str], str]] = None):
        self.name = name
        self.description = description
        self.path = path
        self.post_processing_fn = post_processing_fn


# Define the available tools
available_tools = [
    Tool(
        name="choose",
        description="Choose models to use. Usage: choose(<state>)",
        path="bin/choose.exec", # Assumes the executable is in the current directory
        post_processing_fn=lambda out: int(out)
    ),
    Tool(
        name="blackscholes",
        description="Calculate the price of options. Usage: blackscholes(1, <input_path>, <output_path>)",
        path="bin/blackscholes.exec",
        post_processing_fn=truncate_fn
    ), 
    Tool(
        name="kmeans",
        description="Find K centriods of N nodes (assuming input and output has been taken care of). Usage kmeans(-k, <number of clusers>, -n, <number of nodes>)",
        path="bin/kmeans.exec",
        post_processing_fn=truncate_fn
    ),
    Tool(
        name="lavaMD",
        description="find molecule movement of N random particles. Usage lavaMD(-boxes1d, <number of particles>)",
        path="bin/lavaMD.exec",
        post_processing_fn=truncate_fn
    ),
    Tool(
        name="pagerank.exec",
        description="rank the web-pages of N websites. Usage pagerank(-n, <number of pages>)",
        path="bin/pagerank.exec",
        post_processing_fn=truncate_fn
    )
]


class LLMManager:
    """
    A class to manage the Gemma model and sampler, ensuring it's loaded only once.
    This encapsulates the model inference logic as requested, so the backend
    can be easily swapped with a compiled IREE artifact.
    """
    def __init__(self, model_size: str = "270m"):
        self.model = None
        self.params = None
        self.sampler = None
        self.model_size = model_size
        self._load_model()

    def _load_model(self):
        """Loads the Gemma model and parameters based on the specified size."""
        try:
            if self.model_size == "270m":
                print("Loading Gemma 270M model architecture...")
                self.model = gm.nn.Gemma3_270M()
                print("Loading 270M model parameters...")
                self.params = gm.ckpts.load_params(
                    gm.ckpts.CheckpointPath.GEMMA3_270M_IT
                )
            elif self.model_size == "1b":
                print("Loading Gemma 1B model architecture...")
                self.model = gm.nn.Gemma3_1B()
                print("Loading 1B model parameters...")
                self.params = gm.ckpts.load_params(
                    gm.ckpts.CheckpointPath.GEMMA3_1B_IT
                )
            else:
                # As a fallback or for other models like 4B from the notebook
                print(f"Loading Gemma {self.model_size} model architecture...")
                self.model = gm.nn.Gemma3_4B()
                self.params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

            print("Model and parameters loaded successfully.")

            # The ChatSampler is the easiest way to prompt the model, handling
            # conversation formatting automatically.
            self.sampler = gm.text.ChatSampler(
                model=self.model,
                params=self.params,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please ensure you have authenticated with Kaggle and have the necessary permissions.")
            raise

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the LLM given a prompt.
        This is the core function to be replaced by your compiled artifact's inference call.
        """
        if not self.sampler:
            raise Exception("Model sampler not initialized.")
        print("\n--- Generating LLM Response ---")
        print(f"PROMPT:\n{prompt}")
        print("------------------------------")
        # Using multi_turn=False to ensure each generation is independent
        reply = self.sampler.chat(prompt, multi_turn=False, print_stream=False)
        return reply



class Question:
    """
    [cite_start]Describes a question and its specific function for computing accuracy. [cite: 390]
    """
    def __init__(self, text: str, accuracy_fn: Callable[[str], float]):
        self.text = text
        self.accuracy_fn = accuracy_fn


def tool_invocation(llm_output: str, tools: List[Tool]) -> str:
    """
    Parses the LLM output to find a tool command, invokes the tool,
    [cite_start]and returns its post-processed output. [cite: 393]
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
        return f"Error executing tool '{tool_name}': {e.stderr}"
    except Exception as e:
        return f"An unexpected error occurred during tool invocation: {e}"


def organize_and_run_agent(question: Question, tools: List[Tool], llm: LLMManager) -> Tuple[str, float]:
    """
    [cite_start]Orchestrates the entire agent workflow. [cite: 392]
    1.  LLM selects a tool.
    2.  The tool is invoked.
    3.  LLM generates a final answer based on the tool's output.
    4.  The answer's accuracy is calculated.
    """
    # Step 1: Create a prompt for the LLM to choose a tool.
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    tool_selection_prompt = f"""
You have access to the following tools:
{tool_descriptions}

Your task is to answer the user's question. First, decide if a tool is needed.
If so, respond with ONLY the command to call the tool, like `<tool_name>(arg1, arg2)`.
If no tool is needed, respond with "No tool needed".

Question: {question.text}
"""

    # Ask LLM to select a tool
    llm_tool_choice = llm.generate(tool_selection_prompt)
    print(f"\nLLM Tool Choice: {llm_tool_choice}")

    # Step 2: Invoke the tool
    tool_output = tool_invocation(llm_tool_choice, tools)
    print(f"Processed Tool Output: {tool_output}")

    # Step 3: Generate the final answer using the tool's context
    final_answer_prompt = f"""
You have received the following information from a tool: "{tool_output}"

Based on this information, please provide a clear and concise final answer to the original question.

Original Question: {question.text}
Final Answer:
"""
    
    final_answer = llm.generate(final_answer_prompt)
    print(f"\nFinal Generated Answer: {final_answer}")

    # Step 4: Compute accuracy
    accuracy = question.accuracy_fn(final_answer)
    print(f"Computed Accuracy: {accuracy:.2f}")
    
    return final_answer, accuracy


if __name__ == "__main__":
    llm_manager = LLMManager(model_size="270m")

    # Define a list of questions and their accuracy functions
    # The accuracy function checks if the expected tool command is in the final answer.
    questions_to_run = [
        # --- Blackscholes Questions ---
        Question(
            text="Calculate the prices for the options listed in 'input/in_4.txt' and write the results to 'output/out.txt'. What is the average prices of all the options?",
            accuracy_fn=lambda a: 1.0 if 'blackscholes(1, input/in_4.txt, output/out.txt)' in a.replace(" ", "") else 0.0
        ),
        Question(
            text="I need to price a batch of 16 financial options from 'input/in_16.txt' and save them. What is the maximum price of the first 10 options?",
            accuracy_fn=lambda a: 1.0 if 'blackscholes(1,input/in_16.txt,output/out_16.txt)' in a.replace(" ", "") else 0.0
        ),
        # --- KMeans Questions ---
        Question(
            text="I have 1000 data points that I need to group into 10 distinct clusters. Provide the command to perform this task.",
            accuracy_fn=lambda a: 1.0 if 'kmeans(-k,10,-n,1000)' in a.replace(" ", "") else 0.0
        ),
        Question(
            text="What is the correct tool command to find 5 centroids from a dataset of 500 nodes?",
            accuracy_fn=lambda a: 1.0 if 'kmeans(-k,5,-n,500)' in a.replace(" ", "") else 0.0
        ),
        # --- LavaMD Questions ---
        Question(
            text="Please provide the command to run a molecular dynamics simulation for a system containing 12 particles.",
            accuracy_fn=lambda a: 1.0 if 'lavaMD(-boxes1d,12)' in a.replace(" ", "") else 0.0
        ),
        Question(
            text="How would I simulate the molecular movement of 20 random particles using the available tools?",
            accuracy_fn=lambda a: 1.0 if 'lavaMD(-boxes1d,20)' in a.replace(" ", "") else 0.0
        ),
        # --- PageRank Questions ---
        Question(
            text="I need to calculate the PageRank for a network of 500 web pages. What is the command?",
            accuracy_fn=lambda a: 1.0 if 'pagerank(-n,500)' in a.replace(" ", "") else 0.0
        ),
        Question(
            text="What is the tool invocation required to rank 2000 websites based on their importance in a network?",
            accuracy_fn=lambda a: 1.0 if 'pagerank(-n,2000)' in a.replace(" ", "") else 0.0
        ),
    ]

    # Loop through and process each question
    for i, question in enumerate(questions_to_run):
        print(f"\n\n===== Processing Question {i+1}: '{question.text}' =====")
        organize_and_run_agent(question, available_tools, llm_manager)