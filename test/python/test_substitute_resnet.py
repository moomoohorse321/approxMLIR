from substitute import FuncSubstitute, get_approx_kernel
import tensorflow as tf
import matplotlib.pyplot as plt

#############################################
# Below are testing functions
#############################################

# Static shape, including batch size (1)
INPUT_SHAPE = [1, 224, 224, 3]

# Configuration for our approx kernel
BATCH_SIZE = 32
FEATURES_SHAPE = [224, 224, 3]  # Single image shape (without batch)

def load_image(path_to_image):
    """Load and preprocess an image for ResNet50."""
    image = tf.io.read_file(path_to_image)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image[tf.newaxis, :]
    return image

def decode_result(result):
    """Decode ResNet50 prediction results."""
    return tf.keras.applications.resnet50.decode_predictions(result, top=3)[0]

# Create a pretrained ResNet model with ImageNet weights (this is our "exact" model)
tf_model = tf.keras.applications.resnet50.ResNet50(
    weights="imagenet", include_top=True, input_shape=tuple(FEATURES_SHAPE))

# Wrap the model in a tf.Module to compile it with IREE (exact module)
class ResNetModule(tf.Module):
    def __init__(self):
        super(ResNetModule, self).__init__()
        self.model = tf_model

    @tf.function(input_signature=[tf.TensorSpec(FEATURES_SHAPE, tf.float32)])
    def predict(self, x):
        """Exact prediction function for ResNet50."""
        batched_x = tf.expand_dims(x, 0)  # Add batch dimension
        batched_res = self.model.call(batched_x, training=False)
        return tf.squeeze(batched_res, 0)  # Remove batch dimension for output
    
def test_comparison(self, test_image_path):
    """
    Compare the exact and approximate models on a test image.
    
    Args:
        test_image_path: Path to a test image
    """
    # Load and preprocess the image
    content_image = load_image(test_image_path)
    input_data = tf.keras.applications.resnet50.preprocess_input(content_image)
    
    # Get exact prediction
    exact_input = tf.squeeze(input_data, 0)  # Remove batch dimension
    exact_result = self.exact_module.predict(exact_input)
    batched_exact_result = tf.expand_dims(exact_result, 0)  # Add back batch dimension for decode_result
    exact_decoded = decode_result(batched_exact_result)
    
    # Get approximate prediction (using our kernel)
    # Note: Our kernel takes unbatched input, so we need to squeeze/unsqueeze
    approx_input = tf.squeeze(input_data, 0)  # Remove batch dimension
    approx_result = self.approx_kernel.approx_predict(approx_input)
    approx_result = tf.expand_dims(approx_result, 0)  # Add back batch dimension for decode_result
    approx_decoded = decode_result(approx_result)
    
    # Display results
    print("\nTest Image:")
    plt.figure(figsize=(6, 6))
    plt.imshow(content_image.numpy().reshape(224, 224, 3) / 255.0)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    print("\nExact Model Prediction:")
    for class_name, desc, score in exact_decoded:
        print(f"{desc} ({class_name}): {score:.4f}")
    
    print("\nApproximate Model Prediction:")
    for class_name, desc, score in approx_decoded:
        print(f"{desc} ({class_name}): {score:.4f}")
    
    # Calculate accuracy metrics
    top1_exact = exact_decoded[0][0]
    top1_approx = approx_decoded[0][0]
    print(f"\nTop-1 Exact: {top1_exact}")
    print(f"Top-1 Approx: {top1_approx}")
    print(f"Top-1 Match: {top1_exact == top1_approx}")
    
    # Calculate KL divergence
    kl_div = tf.keras.losses.KLDivergence()(exact_result, approx_result)
    print(f"KL Divergence: {kl_div:.6f}")
        
def test():
    # Define the input shape
    input_shape = FEATURES_SHAPE
    batch_size = BATCH_SIZE
    
    # Create the exact ResNet module
    exact_module = ResNetModule()
    
    # Test constructing an approximate kernel
    test_kernel = get_approx_kernel(input_shape, batch_size)
    print("Successfully created approximation kernel")
    
    # Create the function substitution handler
    func_sub = FuncSubstitute(
        exact_module=exact_module,
        approx_kernel=test_kernel,
        input_shape=input_shape,
        batch_size=batch_size
    )
    
    func_sub.test_comparison = test_comparison.__get__(func_sub)  # Bind the method to the instance
    
    # Training parameters
    num_samples = 10000  # In a real scenario, you'd want more samples
    epochs = 100  # Similarly, more epochs for better results
    
    # Train the approximate kernel
    func_sub.train_approx(num_samples, epochs)
    
    # Compile the approximate kernel to MLIR
    mlir_path = func_sub.compile()
    
    # Test on a sample image
    content_path = tf.keras.utils.get_file(
        'YellowLabradorLooking_new.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
    )
    func_sub.test_comparison(content_path)
    
    print(f"\nApproximate function substitution test complete.")
    print(f"MLIR bytecode saved to: {mlir_path}")
    
if __name__ == "__main__":
    test()