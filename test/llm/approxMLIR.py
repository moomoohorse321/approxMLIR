import iree.compiler.tf
import iree.runtime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from iree.tf.support import module_utils
from iree import runtime as ireert
from iree.compiler import compile_str
import os


import re

_FUNC_DEF_RE = re.compile(r'(\bfunc\.func(?:\s+\w+)*\s+@)([A-Za-z0-9_.$-]+)')

def _collect_function_names(mlir_text: str) -> list[str]:
    """Return all function names defined via 'func.func ... @name'."""
    return [m.group(2) for m in _FUNC_DEF_RE.finditer(mlir_text)]

def _rename_defs(mlir_text: str, name_map: dict[str, str]) -> str:
    def _sub(m: re.Match) -> str:
        prefix, old = m.group(1), m.group(2)
        return f"{prefix}{name_map.get(old, old)}"
    return _FUNC_DEF_RE.sub(_sub, mlir_text)

def _rename_symbol_uses(mlir_text: str, name_map: dict[str, str]) -> str:
    """
    Rename all '@name' symbol uses to the mapped names,
    but skip 'module @name' if present.
    """
    at_sym_re = re.compile(r'@([A-Za-z0-9_.$-]+)')

    def should_skip(idx: int) -> bool:
        window_start = max(0, idx - 20)
        left = mlir_text[window_start:idx]
        m = re.search(r'([A-Za-z0-9_.-]+)\s*$', left)
        return bool(m and m.group(1) == 'module')

    def _sub(m: re.Match) -> str:
        old = m.group(1)
        if old in name_map and not should_skip(m.start()):
            return '@' + name_map[old]
        return m.group(0)

    return at_sym_re.sub(_sub, mlir_text)

def rename_mlir_functions(mlir_text: str, number: int) -> str:
    """Rename all functions to approx_<name>_<number> and update their uses."""
    func_names = _collect_function_names(mlir_text)
    name_map = {name: f"approx_{name}_{number}" for name in func_names}
    out = _rename_defs(mlir_text, name_map)
    out = _rename_symbol_uses(out, name_map)
    return out


# ---------- merging ----------

def _split_top_level_module(mlir_text: str):
    """
    Extract pieces around the FIRST top-level 'module { ... }' (possibly with
    attributes: 'module attributes {...} { ... }').
    Returns:        (preamble, header, body, postamble)
      - preamble:   text before 'module'
      - header:     'module' and any attributes up to (but not including) the '{' that starts the body
      - body:       the text inside the outermost module braces
      - postamble:  anything after the closing '}'
    Raises ValueError if no module block is found.
    """
    # Find the 'module' keyword:
    m = re.search(r'\bmodule\b', mlir_text)
    if not m:
        raise ValueError("No top-level 'module' op found.")

    mod_start = m.start()

    # Find the first '{' after 'module' keyword — that begins the module body.
    brace_open = mlir_text.find('{', m.end())
    if brace_open == -1:
        raise ValueError("Malformed module: missing '{' after 'module'.")

    # Match the corresponding closing brace with a simple brace counter.
    depth = 0
    close_idx = None
    for i in range(brace_open, len(mlir_text)):
        c = mlir_text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                close_idx = i
                break

    if close_idx is None:
        raise ValueError("Malformed module: unbalanced braces.")

    preamble = mlir_text[:mod_start]
    header = mlir_text[mod_start:brace_open].rstrip()
    body = mlir_text[brace_open + 1:close_idx]
    postamble = mlir_text[close_idx + 1:]
    return preamble, header, body, postamble

def merge_mlir_modules(mod1_text: str, mod2_text: str) -> str:
    """
    Merge two MLIR modules into a single module:
      - Uses the exact module header/attributes from mod1
      - Concatenates mod1 body + mod2 body
      - Preserves any preamble before 'module' in mod1 and discards mod2's preamble/postamble
      - If you'd like to keep dialect/preamble lines from mod2, do that before calling this.
    Assumes each input contains ONE top-level 'module'.
    """
    pre1, header1, body1, post1 = _split_top_level_module(mod1_text)
    _,     _,       body2, _     = _split_top_level_module(mod2_text)

    merged_body = f"{body1.strip()}\n\n{body2.strip()}\n"
    merged = f"{pre1}{header1} {{\n{merged_body}}}\n"
    # If mod1 had content after the module (rare), preserve it:
    if post1.strip():
        merged += post1
    return merged





class ToolBox:
    def __init__(self, replace_exec_path, merge_exec_path, opt_exec_path):
        self.replace_exec_path = replace_exec_path
        self.merge_exec_path = merge_exec_path
        self.opt_exec_path = opt_exec_path
    
    @staticmethod
    def load_mlir_from_file(mlir_path, backend_name = "cpu"):
        if backend_name == "cpu":        
            target_backends = ["llvm-cpu"]
        elif backend_name == "gpu":
            target_backends = ["cuda"]
        else:
            raise ValueError("Unsupported backend name. Use 'cpu' or 'gpu'.")
        
        binary_path = mlir_path + ".bin"
        
        if os.path.exists(binary_path):
            print(f"✅ Loading compiled flatbuffer from cache: {binary_path}")
            with open(binary_path, "rb") as f:
                flatbuffer_blob = f.read()
        else:
            print(f"⏳ Compiling MLIR file, no cache found for: {mlir_path}")
            with open(mlir_path, "r") as f:
                mlir_module = f.read()

            # This is the high-overhead compilation step.
            flatbuffer_blob = compile_str(
                mlir_module,
                target_backends=target_backends,
                input_type="stablehlo"
            )

            # Save the compiled flatbuffer to the .bin file for future runs.
            with open(binary_path, "wb") as f:
                print(f"Writing compiled flatbuffer to cache: {binary_path}")
                f.write(flatbuffer_blob)

        config = ireert.Config(target_backends[0])
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer_blob)
        ctx.add_vm_module(vm_module)
        
        return ctx.modules
    
    def link_mlir_modules(self, mlir_path1, mlir_path2, output_path, keep_temp_files=False):
        """
        1. Change 1's name space
        2. merge 1 to 2
        """
        os.system(f"{self.replace_exec_path} @vars. @replace. {mlir_path1} > {mlir_path1}.tmp")
        os.system(f"cp {mlir_path2} {mlir_path2}.tmp")
        os.system(f"{self.merge_exec_path} {mlir_path1}.tmp {mlir_path2}.tmp > {output_path}")
        if not keep_temp_files:
            os.system(f"rm {mlir_path1}.tmp")
            os.system(f"rm {mlir_path2}.tmp")
            
            

     
            
if __name__ == "__main__":
    replace_exec_path = "../../external-tools/approx/replace"
    merge_exec_path = "../../external-tools/approx/merge"
    opt_exec_path = "../../build/bin/approxMLIR-opt"
    mlir_path1 = "./approx.mlir"
    mlir_path2 = "./exact.mlir"
    output_path = "./merged.mlir"
    toolbox = ToolBox(replace_exec_path, merge_exec_path, opt_exec_path)
    toolbox.write2file_auxiliary_mlir_str("./auxiliary.mlir")
    toolbox.link_mlir_modules("./auxiliary.mlir", mlir_path1, "./ext.mlir", keep_temp_files=True)
    toolbox.link_mlir_modules("./ext.mlir", mlir_path2, output_path, keep_temp_files=True)
    toolbox.optimize_mlir(output_path, "./output.mlir")
    
    
# ---------- quick example ----------

# if __name__ == "__main__":
#     # Example usage
#     example = '''
# module {
#   func.func @main(%arg0: tensor<1x4xi32>) -> tensor<1x4xi32> {
#     %0 = call @helper(%arg0) : (tensor<1x4xi32>) -> tensor<1x4xi32>
#     return %0 : tensor<1x4xi32>
#   }
#   func.func @helper(%x: tensor<1x4xi32>) -> tensor<1x4xi32> {
#     // ...
#     return %x : tensor<1x4xi32>
#   }
# }
# '''.strip()

#     renamed = rename_mlir_functions(example, number=7)
#     print(renamed)

#     # Merging (after renaming both sides):
#     modA = rename_mlir_functions(example, number=1)
#     modB = rename_mlir_functions(example, number=2)
#     merged = merge_mlir_modules(modA, modB)
#     print(merged)