import iree.compiler.tf
import iree.runtime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from iree.tf.support import module_utils
from iree import runtime as ireert
from iree.compiler import compile_str
import os


class ToolBox:
    def __init__(self, replace_exec_path, merge_exec_path, opt_exec_path):
        self.replace_exec_path = replace_exec_path
        self.merge_exec_path = merge_exec_path
        self.opt_exec_path = opt_exec_path
    
    @staticmethod
    def load_mlir_from_file(mlir_path):
        backend_choice = "iree_llvmcpu (CPU)" #@param [ "iree_vmvx (CPU)", "iree_llvmcpu (CPU)", "iree_vulkan (GPU/SwiftShader)" ]
        backend_choice = backend_choice.split(" ")[0]
        backend = module_utils.BackendInfo(backend_choice)
        print("Backend choice:", backend_choice)
        
        with open(mlir_path, "r") as f:
            mlir_module = f.read()

        flatbuffer_blob = compile_str(mlir_module, target_backends=["llvm-cpu"], input_type="stablehlo")

        config = ireert.Config(backend.driver)
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer_blob)
        ctx.add_vm_module(vm_module)
        
        return ctx.modules.module
    
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
            
    def optimize_mlir(self, mlir_path, output_path):
        os.system(f"{self.opt_exec_path} {mlir_path} -emit-approx -config-approx > {output_path}")
            
    def write2file_auxiliary_mlir_str(self, mlir_path):
        s = """
            module {
                "approxMLIR.util.annoatation.func_substitution"() <{from = "predict", to = "approx_predict"}> : () -> ()
                "approxMLIR.util.annoatation.func_substitution"() <{from = "learn", to = "approx_learn"}> : () -> ()
            }
        """
        with open(mlir_path, "w") as f:
            f.write(s)
        return s
            
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
    
    