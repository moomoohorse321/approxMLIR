#@title Imports and common setup
import iree.compiler.tf
import iree
from iree import runtime as ireert
from iree.tf.support import module_utils
from iree.compiler import compile_str
import tensorflow as tf
from matplotlib import pyplot as plt

#@title Construct a pretrained ResNet model with ImageNet weights

# Static shape, including batch size (1).
# Can be dynamic once dynamic shape support is ready.
INPUT_SHAPE = [1, 224, 224, 3]

tf_model = tf.keras.applications.resnet50.ResNet50(
    weights="imagenet", include_top=True, input_shape=tuple(INPUT_SHAPE[1:]))

# Wrap the model in a tf.Module to compile it with IREE.
class ResNetModule(tf.Module):

  def __init__(self):
    super(ResNetModule, self).__init__()
    self.model = tf_model

  @tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32)])
  def predict(self, x):
    return self.model.call(x, training=False)

#@markdown ### Backend Configuration

backend_choice = "iree_llvmcpu (CPU)" #@param [ "iree_vmvx (CPU)", "iree_llvmcpu (CPU)", "iree_vulkan (GPU/SwiftShader)" ]
backend_choice = backend_choice.split(" ")[0]
backend = module_utils.BackendInfo(backend_choice)


exported_names = ["predict"]

compiler_module = iree.compiler.tf.compile_module(
    ResNetModule(),
    exported_names=exported_names,
    import_only=True)

# imported_mlirbc_path = "resnet.mlirbc"
# with open(imported_mlirbc_path, "wb") as output_file:
#   output_file.write(compiler_module)
# print(f"Wrote MLIR to path '{imported_mlirbc_path}'")

flatbuffer_blob = compile_str(compiler_module, target_backends=["llvm-cpu"], input_type="stablehlo")

config = ireert.Config(backend.driver)
ctx = ireert.SystemContext(config=config)
vm_module = ireert.VmModule.from_flatbuffer(ctx.instance, flatbuffer_blob)
ctx.add_vm_module(vm_module)

def load_image(path_to_image):
  image = tf.io.read_file(path_to_image)
  image = tf.image.decode_image(image, channels=3)
  image = tf.image.resize(image, (224, 224))
  image = image[tf.newaxis, :]
  return image

content_path = tf.keras.utils.get_file(
    'YellowLabradorLooking_new.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
content_image = load_image(content_path)

print("Test image:")
plt.imshow(content_image.numpy().reshape(224, 224, 3) / 255.0)
plt.axis("off")
plt.tight_layout()
# plt.show()

input_data = tf.keras.applications.resnet50.preprocess_input(content_image)

def decode_result(result):
  return tf.keras.applications.resnet50.decode_predictions(result, top=3)[0]

print("TF prediction:")
tf_result = tf_model.predict(input_data)
print(decode_result(tf_result))

iree_module = ctx.modules.module["predict"]

# print(iree_module)

print("IREE prediction:")
iree_result = iree_module(input_data)
print(iree_result)
print(decode_result(iree_result))