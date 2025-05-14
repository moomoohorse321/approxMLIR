import iree.compiler.tf
import iree.runtime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from iree.tf.support import module_utils
from iree import runtime as ireert
from iree.compiler import compile_str

################################
# Function Substitution Library
################################

# Create our approximate kernel for function substitution
# Create our approximate kernel for function substitution
def get_approx_kernel(input_shape, output_shape, batch_size=32, approx_kernel_func = None, return_type = "NN4Func", nn_params = 128):
    """
        Function to create and return an approximate kernel module.
        User can provide a custom kernel function or use a default NN.
    """
    class ApproxKernel(tf.Module):
        def __init__(self):
            super().__init__()
            
            self.inputs = tf.keras.layers.Input(input_shape)
            
            
            if approx_kernel_func is not None:
                self.outputs = approx_kernel_func(self.inputs)
                
            self.kernel = approx_kernel_func
            
        @tf.function(input_signature=[
            tf.TensorSpec(input_shape, tf.float32) 
        ])
        def approx_predict(self, inputs):
            result = self.kernel(inputs)
            return result
    
    class ApproxNN(ApproxKernel):
        def __init__(self):
            super().__init__()
            if approx_kernel_func is None:
                x = tf.keras.layers.Flatten()(self.inputs)
                x = tf.keras.layers.Dense(nn_params)(x)
                x = tf.keras.layers.Activation("relu")(x)
                if len(output_shape) == 0:
                    x = tf.keras.layers.Dense(1)(x)
                    self.outputs = tf.keras.layers.Activation("sigmoid")(x)
                elif len(output_shape) == 1:
                    x = tf.keras.layers.Dense(output_shape[0])(x) 
                    self.outputs = tf.keras.layers.Softmax()(x)
                else:
                    x = tf.keras.layers.Dense(output_shape[0])(x)
                    x = tf.keras.layers.Activation("relu")(x)
                    for i in range(1, len(output_shape)):
                        x = tf.keras.layers.Dense(output_shape[i])(x)
                        if i < len(output_shape) - 1:
                            x = tf.keras.layers.Activation("relu")(x)
                        else:
                            x = tf.keras.layers.Activation("sigmoid")(x)
                    self.outputs = x
        
            self.model = tf.keras.Model(self.inputs, self.outputs)
            
            # Define loss function and optimizer for training
            self.loss = tf.keras.losses.KLDivergence()
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            
        @tf.function(input_signature=[
            tf.TensorSpec(input_shape, tf.float32)  # Single image (non-batched)
        ])
        def approx_predict(self, inputs):
            """Approximate prediction function for a single image."""
            batched_inputs = tf.expand_dims(inputs, 0)
            result = self.model(batched_inputs, training=False)
            return tf.squeeze(result, 0)
        
        @tf.function(input_signature=[
            tf.TensorSpec([batch_size] + input_shape, tf.float32),  # Batched inputs
            tf.TensorSpec([batch_size] + output_shape, tf.float32)  # Labels (exact model outputs)
        ])
        def approx_learn(self, inputs, labels):
            """Train the approximate kernel to mimic the exact model."""
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss = self.loss(labels, predictions)
                
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss
        
    if return_type == "NN4Func":
        return ApproxNN()
    else:
        return ApproxKernel()

class SubstituteImpl:
    def __init__(self, approx_kernel, exact_module, func_name, input_shape, batch_size=32):
        """
        SubstituteImpl is a class that substitutes an exact function with an approximate one.
        It generates data and trains the model to approximate the exact function.
        
        Args:
            approx_kernel: The NN to substitute the exact function
            exact_module: The module containing the exact function to be substituted
            func_name: The name of the function to be substituted
            input_shape: The shape of a single input (without batch dimension)
            batch_size: Batch size for training
        """
        self.approx_kernel = approx_kernel
        self.exact_module = exact_module
        self.func_name = func_name
        self.input_shape = input_shape
        self.batch_size = batch_size
    
    def gen_data(self, use_provided = True, user_data = None, num_samples = None):
        """
        Generate data for training the model.
        
        Args:
            num_samples: The number of samples to generate
            
        Returns:
            Tuple of (input_data, exact_outputs)
        """
        # Generate random data (in the appropriate range for ResNet)
        # For ResNet, inputs should be in the range expected by preprocess_input
        if not use_provided:
            data = tf.random.uniform((num_samples,) + tuple(self.input_shape), -1.0, 1.0)
        else:
            if list(user_data.shape[1:]) != list(self.input_shape):
                raise ValueError(f"Provided data shape {user_data.shape[1:]} does not match expected input shape {self.input_shape}")
            data = user_data
            num_samples = data.shape[0]
        
        # Get labels by calling the exact function
        exact_func = getattr(self.exact_module, self.func_name)
        
        # Process in batches to avoid memory issues
        labels = []
        for i in range(0, num_samples, self.batch_size):
            end_idx = min(i + self.batch_size, num_samples)
            batch_data = data[i:end_idx]
            batch_labels = []
            
            # For each sample, get the exact prediction
            for j in range(batch_data.shape[0]):
                sample = batch_data[j]  # Add batch dimension
                exact_output = exact_func(sample)
                batch_labels.append(exact_output)  # Remove batch dimension
                
            labels.extend(batch_labels)
            
        return data, tf.stack(labels)
    
    def train_model(self, train_data, train_labels, epochs=5):
        """
        Train the approximate kernel to mimic the exact function.
        
        Args:
            train_data: Input data for training
            train_labels: Output labels from the exact function
            epochs: Number of training epochs
            
        Returns:
            Trained model
        """
        # Training loop
        print(f"Training approximate kernel for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_loss = 0.0
            batches = 0
            
            # Shuffle data for each epoch
            indices = tf.range(start=0, limit=tf.shape(train_data)[0], dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            shuffled_data = tf.gather(train_data, shuffled_indices)
            shuffled_labels = tf.gather(train_labels, shuffled_indices)
            
            for batch_start in range(0, train_data.shape[0], self.batch_size):
                if batch_start + self.batch_size > train_data.shape[0]:
                    continue
                    
                inputs = shuffled_data[batch_start:batch_start + self.batch_size]
                labels = shuffled_labels[batch_start:batch_start + self.batch_size]
                
                loss = self.approx_kernel.approx_learn(inputs, labels)
                epoch_loss += loss
                batches += 1
                
                if batches % 10 == 0:
                    print(f"\rEpoch {epoch+1}/{epochs}, Batch {batches}, Loss: {loss:.4f}", end="")
            
            avg_loss = epoch_loss / batches if batches > 0 else 0
            print(f"\rEpoch {epoch+1}/{epochs} complete, Average Loss: {avg_loss:.4f}")
            
        return self.approx_kernel

class FuncSubstitute:
    def __init__(self, exact_module, approx_kernel, input_shape, batch_size=32):
        """
        Function substitution class for ResNet prediction.
        
        Args:
            exact_module: The module containing the exact function
            input_shape: The shape of a single input (without batch)
            batch_size: Batch size for training
        """
        self.exact_module = exact_module
        self.approx_kernel = approx_kernel
        self.func_name = "predict"
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.impl = SubstituteImpl(approx_kernel, exact_module, self.func_name, input_shape, batch_size)
    
    def train_approx(self, use_provided=True, user_data=None, num_samples=None, epochs=5):
        """
        Train the approximate kernel.
        
        Args:
            use_provided: Whether to use provided user data for training
            user_data: User-provided data for training (if use_provided is True)
            num_samples: Number of samples to generate for training (if use_provided is False)
            epochs: Number of training epochs
            
        Returns:
            Trained approximate kernel
        """
        if use_provided and user_data is None:
            print("=========== Using provided kernel ===========")
            return self.approx_kernel
        if not use_provided and num_samples is None:
            raise ValueError("num_samples must be provided if use_provided is False")
        if use_provided:
            print("=========== Using provided data ===========")
        else:
            print("=========== Generating data ===========")
        train_data, train_labels = self.impl.gen_data(
            use_provided=use_provided,
            user_data=user_data,
            num_samples=num_samples
        )
        print("Data generation complete.")
        
        model = self.impl.train_model(train_data, train_labels, epochs)
        return model
    
    def compile_approx(self, export_dir=".", exported_names = ["approx_predict", "approx_learn"]):
        """
        Compile the approximate kernel to MLIR and save it.
        
        Args:
            export_dir: Directory to save the compiled model
        """
        backend_choice = "iree_llvmcpu"
        
        print("Compiling approximate kernel...")
        mlir_bc = iree.compiler.tf.compile_module(
            self.approx_kernel,
            target_backends=[backend_choice],
            exported_names=exported_names,
            import_only=True
        )
        
        mlir_path = f"{export_dir}/approx.mlirbc"
        with open(mlir_path, "wb") as f:
            f.write(mlir_bc)
        print(f"Wrote MLIR bytecode to {mlir_path}")
        import os 
        os.system(f"iree-ir-tool copy {mlir_path} -o {export_dir}/approx_resnet.mlir")
        
        return mlir_path

    def compile_exact(self, export_dir=".", exported_names = ["predict", "learn"]):
        """
        Compile the exact module to MLIR and save it.
        
        Args:
            export_dir: Directory to save the compiled model
        """
        backend_choice = "iree_llvmcpu"
        
        print("Compiling exact module...")
        mlir_bc = iree.compiler.tf.compile_module(
            self.exact_module,
            target_backends=[backend_choice],
            exported_names=exported_names,
            import_only=True
        )
        
        mlir_path = f"{export_dir}/exact.mlirbc"
        with open(mlir_path, "wb") as f:
            f.write(mlir_bc)
        print(f"Wrote MLIR bytecode to {mlir_path}")
        
        return mlir_path
    
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