import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
import subprocess

import re

class ApproxMLIRSDK:
    def __init__(self, binary_path, mlir_path, compilation_tool_path):
        self.binary_dir_path = binary_path
        self.mlir_dir_path = mlir_path
        self.compilation_dir_path = compilation_tool_path
        
        # Ensure the binary and MLIR directories exist
        os.makedirs(self.binary_dir_path, exist_ok=True)
        os.makedirs(self.mlir_dir_path, exist_ok=True)
        
    def get_knob_val(self, knob_num, state):
        knob_path = os.path.join(self.binary_dir_path, f'choose_{knob_num}.exec')
        
        command = [knob_path, str(state)]
        print(f"Invoking tool: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        match = re.search(r'\d+$', result.stdout.strip())
        if match:
            return int(match.group(0))
        return 1 # hardcoded for now
    
    def get_mlir_files(self, fname = None):
        """
            When auto-tuner wants to read the configurations, it calls toolbox.get_mlir_files()
            When auto-tuner parses its dict, it calls toolbox.get_mlir_files(tool)
            
            return a list of file paths
        """
        if fname:
            # Return the full path for a specific file
            return os.path.join(self.mlir_dir_path, fname)
        else:
            # Return a list of all .mlir files in the directory
            all_files = os.listdir(self.mlir_dir_path)
            mlir_files = [os.path.join(self.mlir_dir_path, f) for f in all_files if f.endswith('.mlir')]
            return mlir_files
    
    def compile_mlir(self, file_path):
        """
            file_path is the MLIR file path, whose compiled artifact will be moved to <binary directory>/<appropriate binary name>
            The return value is the compiled executable path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Source file not found: {file_path}")
        
        gcc_cmd = [
            f"{self.compilation_dir_path}/build/bin/polygeist-opt",
            self.args.mlir_file,
            "-pre-emit-transform",
            "-emit-approx",
            "-config-approx",
            "-transform-approx",
            "-o",
            f"./test.mlir",
        ]

        compile_result = self.call_program(gcc_cmd)
        assert compile_result["returncode"] == 0

        cmd = [
            f"{self.compilation_dir_path}/build/bin/cgeist",
            f"-resource-dir={self.compilation_dir_path}/llvm-project/build/lib/clang/18",
            "-I",
            f"{self.compilation_dir_path}/tools/cgeist/Test/polybench/utilities",
            "-lm",
            "test.mlir",
            "-import-mlir",
            "-o",
            f"./test.exec",
        ]
        base_name = os.path.basename(file_path)
        exec_name = os.path.splitext(base_name).split('_')[1]
        destination_path = os.path.join(self.binary_dir_path, exec_name)
        
        shutil.move("./test.exec", destination_path)
        return destination_path
    
    def populate_orchestration_knobs(self, num_knobs, template_name="approx_choose.mlir"):
        """
        Populates orchestration knobs by copying a template MLIR file.
        For example, it copies 'approx_choose.mlir' to 'approx_choose_1.mlir', 'approx_choose_2.mlir', etc.

        Args:
            num_knobs (int): The number of knob files to create.
            template_name (str): The name of the template file located in the mlir_dir_path.
        
        Returns:
            list[str]: A list of paths to the newly created MLIR files.
        """
        template_path = os.path.join(self.mlir_dir_path, template_name)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template MLIR file not found: {template_path}")
            
        created_files = []
        base_name, extension = os.path.splitext(template_name)

        for i in range(1, num_knobs + 1):
            new_filename = f"{base_name}_{i}{extension}"
            destination_path = os.path.join(self.mlir_dir_path, new_filename)
            
            shutil.copy(template_path, destination_path)
            created_files.append(destination_path)
        
        return created_files
    
    @staticmethod
    def parse_mlir_annotations(mlir_file_path):
        """
        Parse MLIR file for approxMLIR.util.annotation.decision_tree annotations
        and convert them to a dictionary format
        """
        choice_sites = {}

        with open(mlir_file_path, "r") as f:
            mlir_content = f.read()
            
        mlir_fname = os.path.splitext(os.path.basename(mlir_file_path))[0] 

        # Find all approxMLIR.util.annotation.decision_tree annotations
        pattern = r'"approxMLIR\.util\.annotation\.decision_tree"\(\) <\{([^}]+)\}'
        matches = re.findall(pattern, mlir_content, re.DOTALL)

        for match in matches:
            # Extract function name
            func_name_match = re.search(r'func_name = "([^"]+)"', match)
            if not func_name_match:
                continue
            func_name = func_name_match.group(1)

            # Extract num_thresholds
            num_thresholds_match = re.search(r"num_thresholds = (\d+)", match)
            if not num_thresholds_match:
                continue

            # Extract thresholds_lowers array
            thresholds_lowers_match = re.search(
                r"thresholds_lowers = array<i32: ([^>]+)>", match
            )
            if thresholds_lowers_match:
                thresholds_lowers_str = thresholds_lowers_match.group(1)
                thresholds_lowers = [
                    int(x.strip()) for x in thresholds_lowers_str.split(",")
                ]
            else:
                raise ValueError("ill-formed thresholds_lowers.")

            # Extract thresholds_uppers array
            thresholds_uppers_match = re.search(
                r"thresholds_uppers = array<i32: ([^>]+)>", match
            )
            if thresholds_uppers_match:
                thresholds_uppers_str = thresholds_uppers_match.group(1)
                thresholds_uppers = [
                    int(x.strip()) for x in thresholds_uppers_str.split(",")
                ]
            else:
                raise ValueError("ill-formed thresholds_uppers.")

            # Extract decision_values array
            decision_values_match = re.search(
                r"decision_values = array<i32: ([^>]+)>", match
            )
            if decision_values_match:
                decision_values_str = decision_values_match.group(1)
                decision_values = [
                    int(x.strip()) for x in decision_values_str.split(",")
                ]
            else:
                raise ValueError("ill-formed decision_values.")

            # Create dictionary entries for thresholds
            # Use the actual length of the arrays instead of num_thresholds
            for i in range(len(thresholds_lowers)):
                if i < len(thresholds_uppers):
                    key = f"{mlir_fname}_{func_name}_threshold_{i}"
                    value = [thresholds_lowers[i], thresholds_uppers[i]]
                    choice_sites[key] = value

            # Create dictionary entries for decisions
            # Extract decisions array if it exists
            decisions_match = re.search(r"decisions = array<i32: ([^>]+)>", match)
            if decisions_match:
                decisions_str = decisions_match.group(1)
                decisions = [int(x.strip()) for x in decisions_str.split(",")]

                # Create decision entries with ranges based on decision_values
                for i, decision in enumerate(decisions):
                    key = f"{mlir_fname}_{func_name}_decision_{i}"
                    # Use the full range of decision_values for each decision
                    value = [min(decision_values), max(decision_values)]
                    choice_sites[key] = value
            else:
                raise ValueError("ill-formed decision op.")

        return choice_sites
    
    @staticmethod
    def is_thresholds_ascending(thresholds):
        """
        Check if the thresholds array is in ascending order
        """
        if len(thresholds) <= 1:
            return True

        for i in range(1, len(thresholds)):
            if thresholds[i] <= thresholds[i - 1]:
                return False
        return True
    
    @staticmethod
    def modify_mlir_file(self, key, value, func_name, param_type, param_index, mlir_file_path):
        """
        Modify the MLIR file based on the configuration parameter
        """
        # Read the current MLIR file
        with open(mlir_file_path, "r") as f:
            mlir_content = f.read()

        # Find the specific function annotation
        pattern = rf'"approxMLIR\.util\.annotation\.decision_tree"\(\) <\{{[^}}]*func_name = "{func_name}"[^}}]*\}}'
        match = re.search(pattern, mlir_content, re.DOTALL)
        print("match: ", match)
        if match:
            annotation_content = match.group(0)
            modified = False

            if param_type == "threshold":
                # Modify thresholds array
                # Find the thresholds array for this function
                thresholds_pattern = rf"thresholds = array<i32: ([^>]+)>"
                thresholds_match = re.search(thresholds_pattern, annotation_content)

                if thresholds_match:
                    thresholds_str = thresholds_match.group(1)
                    thresholds = [int(x.strip()) for x in thresholds_str.split(",")]

                    # Update the specific threshold at the given index
                    if param_index < len(thresholds):
                        thresholds[param_index] = value

                        # Check if thresholds array is in ascending order
                        if not self.is_thresholds_ascending(thresholds):
                            print(
                                f"ERROR: Thresholds array is not in ascending order: {thresholds}"
                            )
                            return (
                                False  # Return False to indicate validation failure
                            )

                        new_thresholds_str = ", ".join(map(str, thresholds))

                        # Replace in the annotation content using regex to be more specific
                        # This ensures we only replace the thresholds array in this specific annotation
                        thresholds_regex = (
                            rf"thresholds = array<i32: {re.escape(thresholds_str)}>"
                        )
                        new_thresholds_line = (
                            f"thresholds = array<i32: {new_thresholds_str}>"
                        )
                        annotation_content = re.sub(
                            thresholds_regex,
                            new_thresholds_line,
                            annotation_content,
                        )
                        modified = True

            elif param_type == "decision":
                # Modify decisions array
                decisions_pattern = rf"decisions = array<i32: ([^>]+)>"
                decisions_match = re.search(decisions_pattern, annotation_content)

                if decisions_match:
                    decisions_str = decisions_match.group(1)
                    decisions = [int(x.strip()) for x in decisions_str.split(",")]

                    # Update the specific decision at the given index
                    if param_index < len(decisions):
                        decisions[param_index] = value
                        new_decisions_str = ", ".join(map(str, decisions))

                        # Replace in the annotation content using regex to be more specific
                        # This ensures we only replace the decisions array in this specific annotation
                        decisions_regex = (
                            rf"decisions = array<i32: {re.escape(decisions_str)}>"
                        )
                        new_decisions_line = (
                            f"decisions = array<i32: {new_decisions_str}>"
                        )
                        annotation_content = re.sub(
                            decisions_regex, new_decisions_line, annotation_content
                        )
                        modified = True

            if modified:
                # Replace the entire annotation in the MLIR content
                mlir_content = mlir_content.replace(
                    match.group(0), annotation_content
                )

                # Write the modified content back to the file
                with open(mlir_file_path, "w") as f:
                    f.write(mlir_content)

                print(f"Successfully modified MLIR file: {key} = {value}")
            else:
                print(f"Could not modify parameter {key} in MLIR file")
        else:
            print(f"Could not find function {func_name} in MLIR file")

     
            
if __name__ == "__main__":
    pass
    