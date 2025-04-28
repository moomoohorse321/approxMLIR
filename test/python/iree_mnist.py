import iree.compiler.tf
import iree.runtime
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from iree.tf.support import module_utils
from iree import runtime as ireert


from iree_mnist_impl import NUM_CLASSES, NUM_ROWS, NUM_COLS, BATCH_SIZE
from iree_mnist_impl import load_data
from iree.compiler import compile_str

from approxMLIR import approxModule

def get_trainableDNN(batcnh_size, num_rows, num_cols, num_channels):
  class TrainableDNN(tf.Module):

    def __init__(self):
      super().__init__()

      # Create a Keras model to train.
      inputs = tf.keras.layers.Input((num_rows, num_cols, num_channels))
      x = tf.keras.layers.Flatten()(inputs)
      x = tf.keras.layers.Dense(128)(x)
      x = tf.keras.layers.Activation("relu")(x)
      x = tf.keras.layers.Dense(10)(x)
      outputs = tf.keras.layers.Softmax()(x)
      self.model = tf.keras.Model(inputs, outputs)

      # Create a loss function and optimizer to use during training.
      self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
      self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

    @tf.function(input_signature=[
        tf.TensorSpec([batcnh_size, num_rows, num_cols, num_channels])  # inputs
    ])
    def predict(self, inputs):
      return self.model(inputs, training=False)
    

    # We compile the entire training step by making it a method on the model.
    @tf.function(input_signature=[
        tf.TensorSpec([batcnh_size, num_rows, num_cols, num_channels]),  # inputs
        tf.TensorSpec(batcnh_size, tf.int32)  # labels
    ])
    def learn(self, inputs, labels):
      # Capture the gradients from forward prop...
      # achor = self.start_knob()
      # with tf.control_dependencies([achor]):
      with tf.GradientTape() as tape:
        """
          One way to make approxMLIR neat is function-level annotation / rewrite. The core idea is to decompose larger functions into smaller ones. The following is an explanation to justify why this decomposition won't work:
          The problem is that whenever you take out the function, it will get inlined anyway for python (compute graph) before you get the chance to optimize itself.
          Then you probably will try to only compile the small functions, without giving them the context. It still won't work.
          As an example, we can see here self.model isn't a functional function. It will impact the gradient tape. So we can't compile self.model alone, at least not without its context.
          This enforces us to drop any annotations (will explain later), also working at a sub-graph scope instead of function scope.
          The example is training, but the same reasoning is for re-inforcement learning.
        """
        probs = self.model(inputs, training=True)
        loss = self.loss(labels, probs)

      # ...and use them to update the model's weights.
      variables = self.model.trainable_variables
      gradients = tape.gradient(loss, variables)
      self.optimizer.apply_gradients(zip(gradients, variables))
      
      return loss
  return TrainableDNN()
  
  
  
  
exported_names = ["predict", "learn"]
backend_choice = "iree_llvmcpu (CPU)" #@param [ "iree_vmvx (CPU)", "iree_llvmcpu (CPU)", "iree_vulkan (GPU/SwiftShader)" ]
backend_choice = backend_choice.split(" ")[0]
backend = module_utils.BackendInfo(backend_choice)

print("Backend choice:", backend_choice)
backend = module_utils.BackendInfo(backend_choice)

mlir_bc = iree.compiler.tf.compile_module(
    get_trainableDNN(
        BATCH_SIZE, NUM_ROWS, NUM_COLS, 1),
    target_backends=["llvm-cpu"],
    exported_names=exported_names,
    import_only=True)

with open("lenet.mlirbc", "wb") as f:
    f.write(mlir_bc)

# with open("lenet.mlir", "r") as f:
#   mlir_module = f.read()

# flatbuffer_blob = compile_str(mlir_bc, target_backends=["llvm-cpu"], input_type="stablehlo")

# config = ireert.Config(backend.driver)
# ctx = ireert.SystemContext(config=config)
# vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer_blob)
# ctx.add_vm_module(vm_module)


# iree_predict = ctx.modules.module["predict"]
# iree_learn = ctx.modules.module["learn"]

# (x_train, y_train), (x_test, y_test) = load_data()
# #@title Benchmark inference and training
# iree_predict(x_train[:BATCH_SIZE])
# print("loss:", iree_learn(x_train[:BATCH_SIZE], y_train[:BATCH_SIZE]).to_host())


# losses = []

# step = 0
# max_steps = x_train.shape[0] // BATCH_SIZE

# for batch_start in range(0, x_train.shape[0], BATCH_SIZE):
#   if batch_start + BATCH_SIZE > x_train.shape[0]:
#     continue

#   inputs = x_train[batch_start:batch_start + BATCH_SIZE]
#   labels = y_train[batch_start:batch_start + BATCH_SIZE]

#   loss = iree_learn(inputs, labels).to_host()
#   losses.append(loss)

#   step += 1
#   print(f"\rStep {step:4d}/{max_steps}: loss = {loss:.4f}", end="")