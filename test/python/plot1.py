import iree.compiler.tf
import iree.runtime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from iree.tf.support import module_utils
from iree import runtime as ireert
from iree.compiler import compile_str
from substitute import FuncSubstitute, get_approx_kernel
import time

# Configuration for our MNIST model
NUM_CLASSES = 10
NUM_ROWS, NUM_COLS = 28, 28
BATCH_SIZE = 32
INPUT_SHAPE = [1, NUM_ROWS, NUM_COLS, 1]  # Static shape with batch size of 1
OUTPUT_SHAPE = [NUM_CLASSES]  # Static shape for output (batch size of 1)
FEATURES_SHAPE = [NUM_ROWS, NUM_COLS, 1]  # Single image shape (without batch)

def load_data():
    """Load MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape into grayscale images
    x_train = np.reshape(x_train, (-1, NUM_ROWS, NUM_COLS, 1))
    x_test = np.reshape(x_test, (-1, NUM_ROWS, NUM_COLS, 1))

    # Rescale pixel values to [0, 1]
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    # Convert labels to one-hot encoding for the classifier
    y_train_onehot = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_onehot = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot)

# Create a CNN model for MNIST digit recognition (this is our "exact" model)
def create_mnist_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=FEATURES_SHAPE),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Wrap the model in a tf.Module to compile it with IREE
def create_mnist_module(batch_size=BATCH_SIZE):
    class MNISTModule(tf.Module):
        def __init__(self):
            super(MNISTModule, self).__init__()
            self.model = create_mnist_model()
            
            # Compile the model
            self.model.compile(
                optimizer='adam',
                loss=tf.keras.losses.KLDivergence(),
                metrics=['accuracy']
            )

        @tf.function(input_signature=[tf.TensorSpec(FEATURES_SHAPE, tf.float32)])
        def predict(self, x):
            """Exact prediction function for MNIST. (non-batched)"""
            batched_x = tf.expand_dims(x, 0)  # Add batch dimension
            batched_res = self.model(batched_x, training=False)
            return tf.squeeze(batched_res, 0)  # Remove batch dimension for output
        
        @tf.function(input_signature=[
            tf.TensorSpec([batch_size] + FEATURES_SHAPE, tf.float32),
            tf.TensorSpec([batch_size, NUM_CLASSES], tf.float32)  # One-hot encoded labels
        ])
        def learn(self, x, y):
            """Train the model on batched data."""
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=True)
                loss = tf.keras.losses.KLDivergence()(y, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return loss
        
    return MNISTModule()

def train_exact_module(model, data, epochs=5):
    """Train the trainable model on MNIST data with real-time line updates."""
    (x_train, y_train, y_train_onehot) = data
    
    # Set up training loop
    steps_per_epoch = len(x_train) // BATCH_SIZE
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            # Get batch
            batch_start = step * BATCH_SIZE
            batch_end = batch_start + BATCH_SIZE
            x_batch = x_train[batch_start:batch_end]
            y_batch = y_train_onehot[batch_start:batch_end] 
            
            # Perform one training step
            step_loss = model.learn(x_batch, y_batch)
            epoch_loss += step_loss
            
            # Update progress line (overwrites previous line)
            if step % 10 == 0:
                print(f"\rEpoch {epoch+1}/{epochs}, Step {step}/{steps_per_epoch}", end="")
        
        # Print epoch summary (overwrites previous line)
        print(f"\rEpoch {epoch+1}/{epochs} complete, Average Loss: {epoch_loss / steps_per_epoch}", end="")
    print("\nTraining complete.")
    
    return model

def test_comparison(self, test_images, test_labels, num_samples=10, use_mlir_approx=True):
    """
    Compare the exact and approximate models on test images.
    
    Args:
        test_images: Test images
        test_labels: Test labels
        num_samples: Number of samples to test
    """
    # Select random samples for testing
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    
    correct_exact = 0
    correct_approx = 0
    avg_kl_div = 0.0
    
    plt.figure(figsize=(15, num_samples * 2))
    
    for i, idx in enumerate(indices):
        # Get test image and label
        test_image = test_images[idx]
        true_label = test_labels[idx]
        
        # Get exact prediction
        exact_result = self.exact_module.predict(test_image)
        exact_pred = np.argmax(exact_result.numpy())
        
        # Get approximate prediction
        if use_mlir_approx:
            approx_result = self.approx_kernel.approx_predict(test_image).to_host()
        else:
            approx_result = self.approx_kernel.approx_predict(test_image).numpy()
        approx_pred = np.argmax(approx_result)
        
        # Update counters
        if exact_pred == true_label:
            correct_exact += 1
        if approx_pred == true_label:
            correct_approx += 1
        
        # Calculate KL divergence
        kl_div = tf.keras.losses.KLDivergence()(exact_result, approx_result)
        avg_kl_div += kl_div.numpy()
        
        # Display image and predictions
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(test_image.reshape(NUM_ROWS, NUM_COLS), cmap='gray')
        plt.title(f"True: {true_label}")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.bar(range(NUM_CLASSES), exact_result.numpy())
        plt.title(f"Exact: {exact_pred}" + (" ✓" if exact_pred == true_label else " ✗"))
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.bar(range(NUM_CLASSES), approx_result)
        plt.title(f"Approx: {approx_pred}" + (" ✓" if approx_pred == true_label else " ✗"))
    
    plt.tight_layout()
    plt.savefig("figure1.png")
    plt.show()
    
    # Print results
    print(f"\nExact Model Accuracy: {correct_exact / num_samples:.2f}")
    print(f"Approximate Model Accuracy: {correct_approx / num_samples:.2f}")
    print(f"Average KL Divergence: {avg_kl_div / num_samples:.6f}")
    
    # Test on full test set
    exact_correct = 0
    approx_correct = 0
    
    for i in range(len(test_images)):
        # Get test image and label
        test_image = test_images[i]
        true_label = test_labels[i]
        
        # Get exact prediction
        exact_result = self.exact_module.predict(test_image)
        exact_pred = np.argmax(exact_result.numpy())
        
        # Get approximate prediction
        if use_mlir_approx:
            approx_result = self.approx_kernel.approx_predict(test_image).to_host()
        else:
            approx_result = self.approx_kernel.approx_predict(test_image).numpy()
        approx_pred = np.argmax(approx_result)
        
        # Update counters
        if exact_pred == true_label:
            exact_correct += 1
        if approx_pred == true_label:
            approx_correct += 1
            
    print(f"\nFull Test Set - Exact Model Accuracy: {exact_correct / len(test_images):.4f}")
    print(f"Full Test Set - Approximate Model Accuracy: {approx_correct / len(test_images):.4f}")

def test():
    # Load MNIST data
    exact_module_path = "mnist_exact_model"
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot) = load_data()
    
    # # Create and train the exact MNIST module
    # exact_module = create_mnist_module(BATCH_SIZE)
    # print("Training the exact model...")
    # exact_module = train_exact_module(exact_module, (x_train, y_train, y_train_onehot), epochs=5)
    # print("Exact model training complete.")
    
    # # # save it to .pth
    # tf.saved_model.save(exact_module, exact_module_path)
    # print(f"Exact model saved to {exact_module_path}")
    
    # load it back
    exact_module = tf.saved_model.load(exact_module_path)
    print("Exact model loaded from saved file.")
    
    # Test constructing an approximate kernel
    print("Creating approximation kernel...")
    test_kernel = get_approx_kernel(FEATURES_SHAPE, OUTPUT_SHAPE, BATCH_SIZE)
    print("Successfully created approximation kernel")
    
    # Create the function substitution handler
    func_sub = FuncSubstitute(
        exact_module=exact_module,
        approx_kernel=test_kernel,
        input_shape=FEATURES_SHAPE,
        batch_size=BATCH_SIZE
    )
    
    func_sub.compile_exact()
    
    func_sub.test_comparison = test_comparison.__get__(func_sub)  # Bind the method to the instance
    
    # Training parameters
    num_samples = 5000  # Number of samples to use for training the approximation
    epochs = 50         # Number of training epochs
    
    # Train the approximate kernel
    print(f"Training approximation kernel with up to {num_samples} samples for {epochs} epochs...")
    print(f"There are {len(x_train)} samples in total.")
    func_sub.train_approx(use_provided=True, user_data=x_train[:num_samples], epochs=epochs)
    print("Approximation training complete.")
    
    # Compile the approximate kernel to MLIR
    mlir_path = func_sub.compile_approx()
    
    # Test on sample images
    print("\nComparing exact and approximate models on test samples...")
    func_sub.test_comparison(x_test, y_test, num_samples=10, use_mlir_approx=False)
    
    print(f"\nApproximate function substitution test complete.")
    print(f"MLIR bytecode saved to: {mlir_path}")
    
    return exact_module, func_sub


def test_comparison_modified(self, test_images, test_labels, num_samples_to_plot=0, use_mlir_approx=False, show_sample_plots=False):
    """
    Compare the exact and approximate models on test images.
    Returns exact and approximate accuracies on the full test set, and optionally performance improvement.
    """
    import time
    if show_sample_plots and num_samples_to_plot > 0 and num_samples_to_plot <= len(test_images):
        indices = np.random.choice(len(test_images), num_samples_to_plot, replace=False)
        correct_exact_sample = 0
        correct_approx_sample = 0
        avg_kl_div_sample = 0.0
        
        plt.figure(figsize=(15, num_samples_to_plot * 3.5))
        for i, idx in enumerate(indices):
            test_image = test_images[idx]
            true_label = test_labels[idx]
            
            exact_result_tf = self.exact_module.predict(test_image)
            exact_result_np = exact_result_tf.numpy()
            exact_pred = np.argmax(exact_result_np)
            
            if use_mlir_approx:
                # This path assumes self.approx_kernel is the compiled/loaded MLIR module interface
                approx_result_host = self.approx_kernel.approx_predict(test_image) #.to_host() might be needed depending on type
                approx_result_np = approx_result_host.to_host() if hasattr(approx_result_host, 'to_host') else approx_result_host
            else:
                # This path assumes self.approx_kernel is a TF model/module with a .predict or .approx_predict method
                approx_result_tf = self.approx_kernel.approx_predict(test_image)
                approx_result_np = approx_result_tf.numpy() if hasattr(approx_result_tf, 'numpy') else approx_result_tf

            approx_pred = np.argmax(approx_result_np)
            
            if exact_pred == true_label: correct_exact_sample += 1
            if approx_pred == true_label: correct_approx_sample += 1
            
            kl_div = tf.keras.losses.KLDivergence()(exact_result_np, approx_result_np)
            avg_kl_div_sample += kl_div.numpy()
            
            plt.subplot(num_samples_to_plot, 3, i*3 + 1)
            plt.imshow(test_image.reshape(NUM_ROWS, NUM_COLS), cmap='gray')
            plt.title(f"True: {true_label}")
            plt.axis('off')
            
            plt.subplot(num_samples_to_plot, 3, i*3 + 2)
            plt.bar(range(NUM_CLASSES), exact_result_np)
            plt.title(f"Exact: {exact_pred}" + (" ✓" if exact_pred == true_label else " ✗"))
            
            plt.subplot(num_samples_to_plot, 3, i*3 + 3)
            plt.bar(range(NUM_CLASSES), approx_result_np)
            plt.title(f"Approx: {approx_pred}" + (" ✓" if approx_pred == true_label else " ✗"))
        
        plt.tight_layout()
        plt.savefig("figure_sample_comparison.png")
        plt.show()
        
        if num_samples_to_plot > 0:
            print(f"\nSampled Test Results ({num_samples_to_plot} samples):")
            print(f"Exact Model Accuracy (Sampled): {correct_exact_sample / num_samples_to_plot:.2f}")
            print(f"Approximate Model Accuracy (Sampled): {correct_approx_sample / num_samples_to_plot:.2f}")
            print(f"Average KL Divergence (Sampled): {avg_kl_div_sample / num_samples_to_plot:.6f}")

    # Test on full test set
    exact_correct_full = 0
    approx_correct_full = 0
    
    # Placeholder for performance timing
    exact_total_time = 0.0
    approx_total_time = 0.0

    num_full_test_samples = len(test_images)
    print(f"Evaluating on full test set of {num_full_test_samples} images...")
    for i in range(num_full_test_samples):
        if (i + 1) % 500 == 0:
            print(f"\rProcessing full test set: {i+1}/{num_full_test_samples}", end="")

        test_image = test_images[i]
        true_label = test_labels[i]
        
        start_exact = time.perf_counter()
        exact_result_tf = self.exact_module.predict(test_image)
        exact_result_np = exact_result_tf.numpy()
        end_exact = time.perf_counter()
        exact_total_time += (end_exact - start_exact)
        exact_pred = np.argmax(exact_result_np)
        
        start_approx = time.perf_counter()
        if use_mlir_approx:
            approx_result_host = self.approx_kernel.approx_predict(test_image)
            approx_result_np = approx_result_host.to_host() if hasattr(approx_result_host, 'to_host') else approx_result_host
        else:
            approx_result_tf = self.approx_kernel.approx_predict(test_image) # Ensure this method exists on your approx_kernel
            approx_result_np = approx_result_tf.numpy() if hasattr(approx_result_tf, 'numpy') else approx_result_tf

        end_approx = time.perf_counter()
        approx_total_time += (end_approx - start_approx)
        approx_pred = np.argmax(approx_result_np)
        
        if exact_pred == true_label: exact_correct_full += 1
        if approx_pred == true_label: approx_correct_full += 1
            
    print(f"\rProcessing full test set: {num_full_test_samples}/{num_full_test_samples}... Done." + " "*10)

    exact_accuracy_full = exact_correct_full / num_full_test_samples if num_full_test_samples > 0 else 0
    approx_accuracy_full = approx_correct_full / num_full_test_samples if num_full_test_samples > 0 else 0
    
    print(f"\nFull Test Set - Exact Model Accuracy: {exact_accuracy_full:.4f}")
    print(f"Full Test Set - Approximate Model Accuracy: {approx_accuracy_full:.4f}")

    performance_improvement = 0.0
    if exact_total_time > 0 and num_full_test_samples > 0: # Avoid division by zero
        avg_exact_time = exact_total_time / num_full_test_samples
        avg_approx_time = approx_total_time / num_full_test_samples
        print(f"Full Test Set - Avg Exact Time per sample: {avg_exact_time*1000:.2f} ms")
        print(f"Full Test Set - Avg Approx Time per sample: {avg_approx_time*1000:.2f} ms")
        performance_improvement = ((avg_exact_time - avg_approx_time) / avg_exact_time) * 100
        print(f"Full Test Set - Performance Improvement (Approx vs Exact): {performance_improvement:.2f}%")
    
    return exact_accuracy_full, approx_accuracy_full , performance_improvement


def run_insight1_experiments():
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot) = load_data()
    
    exact_module_path = "mnist_exact_model"
    
    exact_module = tf.saved_model.load(exact_module_path)
    print("Exact model loaded from saved file.")

    temp_approx_kernel_for_exact_eval = get_approx_kernel(FEATURES_SHAPE, OUTPUT_SHAPE, BATCH_SIZE)
    temp_func_sub_for_exact = FuncSubstitute(
        exact_module=exact_module,
        approx_kernel=temp_approx_kernel_for_exact_eval, # This kernel won't be used for exact part
        input_shape=FEATURES_SHAPE, # Ensure this matches what FuncSubstitute expects
        batch_size=BATCH_SIZE
    )
    if hasattr(temp_func_sub_for_exact, 'compile_exact'): # If compile_exact exists and is needed
        temp_func_sub_for_exact.compile_exact()
        
    temp_func_sub_for_exact.test_comparison = test_comparison_modified.__get__(temp_func_sub_for_exact)
    
    # Get exact_accuracy_baseline. The second value (approx_acc) is from an untrained kernel, so ignore.
    exact_accuracy_baseline, _, _ = temp_func_sub_for_exact.test_comparison(
        x_test, y_test, num_samples_to_plot=0, use_mlir_approx=False, show_sample_plots=False
    )
    print(f"Baseline Exact Model Accuracy (full test set): {exact_accuracy_baseline:.4f}")
    if exact_accuracy_baseline == 0:
        print("ERROR: Baseline exact accuracy is 0. Cannot calculate accuracy drop. Please check the exact model.")
        return

    # --- Experiment 1: Tune number of training samples ---
    print("\n--- Starting Experiment 1: Tuning Number of Training Samples ---")
    num_samples_values = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    # Ensure x_train is defined and loaded
    # num_samples_values = [s for s in num_samples_values if s <= len(x_train)]
    if not num_samples_values:
        print("Warning: No valid number of samples to test for Experiment 1.")
        # return # Or handle appropriately

    fixed_epochs_for_exp1 = 50
    accuracy_drops_exp1 = []
    training_times_exp1 = [] # To store training times

    for n_samples in num_samples_values:
        print(f"\nTraining approx kernel: {n_samples} samples, {fixed_epochs_for_exp1} epochs...")
        # Ensure exact_module, get_approx_kernel, FEATURES_SHAPE, OUTPUT_SHAPE, BATCH_SIZE are defined
        approx_kernel_exp1 = get_approx_kernel(FEATURES_SHAPE, OUTPUT_SHAPE, BATCH_SIZE)
        func_sub_exp1 = FuncSubstitute(exact_module, approx_kernel_exp1, FEATURES_SHAPE, BATCH_SIZE)
        
        if hasattr(func_sub_exp1, 'compile_exact'):
            func_sub_exp1.compile_exact()

        # --- Measure training time ---
        start_time = time.perf_counter()
        # Ensure train_approx and x_train are available
        func_sub_exp1.train_approx(use_provided=True, user_data=x_train[:n_samples], epochs=fixed_epochs_for_exp1)
        end_time = time.perf_counter()
        training_time = end_time - start_time
        training_times_exp1.append(training_time)
        # --- End of training time measurement ---
        
        # Bind the modified test_comparison method
        # Ensure test_comparison_modified is defined
        func_sub_exp1.test_comparison = test_comparison_modified.__get__(func_sub_exp1)
        
        # Assuming test_comparison_modified returns (exact_acc, approx_acc)
        # If it returns performance as a third value, adjust unpacking: _exact_acc, approx_acc, _perf = ...
        # Ensure x_test, y_test are defined
        _exact_acc, approx_acc, _ = func_sub_exp1.test_comparison(
            x_test, y_test, num_samples_to_plot=0, use_mlir_approx=False, show_sample_plots=False
        )
        
        # Ensure exact_accuracy_baseline is a non-zero value
        if exact_accuracy_baseline == 0:
            print("Error: exact_accuracy_baseline is zero, cannot compute accuracy drop. Skipping.")
            accuracy_drop = float('nan') # Or handle as an error
        else:
            accuracy_drop = ((exact_accuracy_baseline - approx_acc) / exact_accuracy_baseline) * 100
        
        accuracy_drops_exp1.append(accuracy_drop)
        print(f"  Approx Acc: {approx_acc:.4f}, Accuracy Drop: {accuracy_drop:.2f}% (vs baseline {exact_accuracy_baseline:.4f})")
        print(f"  Training Time: {training_time:.2f} seconds")

    # --- Modified Plotting for Experiment 1 ---
    fig, ax1 = plt.subplots(figsize=(12, 7)) # Increased figure size for better readability

    # Plot Accuracy Drop on the primary y-axis (left)
    color_acc = 'tab:blue'
    ax1.set_xlabel('Number of Training Samples (Log Scale)')
    ax1.set_ylabel('Accuracy Drop (%)', color=color_acc)
    ax1.plot(num_samples_values, accuracy_drops_exp1, marker='o', linestyle='-', color=color_acc, label='Accuracy Drop')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_xscale('log') # Keep x-axis as log scale
    ax1.grid(True, which="both", ls="-", axis='x') # Grid for x-axis
    ax1.grid(True, which="major", ls="--", axis='y', color=color_acc, alpha=0.7) # Grid for primary y-axis

    # Create a secondary y-axis (right) for Training Time
    ax2 = ax1.twinx()
    color_time = 'tab:red'
    ax2.set_ylabel('Training Time (seconds)', color=color_time)
    ax2.plot(num_samples_values, training_times_exp1, marker='s', linestyle='--', color=color_time, label='Training Time')
    ax2.tick_params(axis='y', labelcolor=color_time)
    # ax2.set_yscale('log') # Optional: if training times also vary over orders of magnitude

    plt.title('Accuracy Drop & Training Time vs. Number of Training Samples')

    # Add legends
    # To combine legends from two axes, you can do it manually:
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right') # Adjust location as needed, e.g., 'best', 'upper left'

    fig.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.savefig("plot_accuracy_drop_and_training_time_vs_samples.png")
    plt.show()
    print("Plot for Experiment 1 saved as plot_accuracy_drop_and_training_time_vs_samples.png")

    # --- Experiment 2: Tune number of epochs ---
    print("\n--- Starting Experiment 2: Tuning Number of Epochs ---")
    epochs_values = list(range(1, 51, 5))
    fixed_samples_for_exp2 = 5000
    if fixed_samples_for_exp2 > len(x_train):
        fixed_samples_for_exp2 = len(x_train)
        print(f"Adjusted fixed samples for Exp2 to {fixed_samples_for_exp2} (max available).")

    accuracy_drops_exp2 = []
    # performance_improvements_exp2 = []

    for n_epochs in epochs_values:
        print(f"\nTraining approx kernel: {fixed_samples_for_exp2} samples, {n_epochs} epochs...")
        approx_kernel_exp2 = get_approx_kernel(FEATURES_SHAPE, OUTPUT_SHAPE, BATCH_SIZE)
        func_sub_exp2 = FuncSubstitute(exact_module, approx_kernel_exp2, FEATURES_SHAPE, BATCH_SIZE)
        if hasattr(func_sub_exp2, 'compile_exact'): func_sub_exp2.compile_exact()
        
        func_sub_exp2.train_approx(use_provided=True, user_data=x_train[:fixed_samples_for_exp2], epochs=n_epochs)
        
        func_sub_exp2.test_comparison = test_comparison_modified.__get__(func_sub_exp2)
        _exact_acc, approx_acc, _ = func_sub_exp2.test_comparison(
            x_test, y_test, num_samples_to_plot=0, use_mlir_approx=False, show_sample_plots=False
        )
        
        accuracy_drop = ((exact_accuracy_baseline - approx_acc) / exact_accuracy_baseline) * 100
        accuracy_drops_exp2.append(accuracy_drop)
        # performance_improvements_exp2.append(perf_improve_val)
        print(f"  Approx Acc: {approx_acc:.4f}, Accuracy Drop: {accuracy_drop:.2f}% (vs baseline {exact_accuracy_baseline:.4f})")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_values, accuracy_drops_exp2, marker='o', linestyle='-')
    plt.title('Accuracy Drop vs. Number of Epochs (Approx. Kernel)')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy Drop (%) [(Exact - Approx) / Exact * 100]')
    plt.grid(True)
    plt.savefig("plot_accuracy_drop_vs_epochs.png")
    print("Plot for Experiment 2 saved as plot_accuracy_drop_vs_epochs.png")


def test_load():
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot) = load_data()
    
    exact_module_path = "mnist_exact_model"
    # load it back
    exact_module = tf.saved_model.load(exact_module_path)
    print("Exact model loaded from saved file.")
    
    # Test constructing an approximate kernel
    print("Creating approximation kernel...")
    test_kernel = get_approx_kernel(FEATURES_SHAPE, OUTPUT_SHAPE, BATCH_SIZE)
    print("Successfully created approximation kernel")
    
    # Create the function substitution handler
    func_sub = FuncSubstitute(
        exact_module=exact_module,
        approx_kernel=FuncSubstitute.load_mlir_from_file("output.mlir"),
        input_shape=FEATURES_SHAPE,
        batch_size=BATCH_SIZE
    )
    
    func_sub.test_comparison = test_comparison.__get__(func_sub)
    
    func_sub.test_comparison(
        x_test, y_test, num_samples=10
    )
    
    

if __name__ == "__main__":
    run_insight1_experiments()