import iree.compiler.tf
import iree.runtime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from iree.tf.support import module_utils
from iree import runtime as ireert
from iree.compiler import compile_str
from substitute import FuncSubstitute, get_approx_kernel
import time, os

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

def post_train_model(model, data, epochs=5):
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
    You need to change this function to do the post-training instead of testing inference. 
    
    Then you must plot a diagram with x-axis being the training progress and y-axis being the loss and the inference time.
    
    It's trivial that there are 2 curves, one for the exact model and one for the approximate model.
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
    exact_module_path = "mnist_exact_for_posttraining"
    (x_train, y_train, y_train_onehot), (x_test, y_test, y_test_onehot) = load_data()
    
    # Create and train the exact MNIST module
    exact_module = create_mnist_module(BATCH_SIZE)
    print("Training the exact model...")
    # We only save a pre-trained model, such that we can do post-training later.
    exact_module = train_exact_module(exact_module, (x_train, y_train, y_train_onehot), epochs=1)
    print("Exact model training complete.")
    
    # save it to .pth
    tf.saved_model.save(exact_module, exact_module_path)
    print(f"Exact model saved to {exact_module_path}")
    
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
    
    
def initial_train_model(model_module, data, epochs=1):
    """Initial training for a model module."""
    (x_train, _y_train_labels, y_train_onehot) = data
    steps_per_epoch = len(x_train) // BATCH_SIZE
    print(f"Starting initial training for for {epochs} epoch(s)...")
    for epoch in range(epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for step in range(steps_per_epoch):
            batch_start = step * BATCH_SIZE
            batch_end = (step + 1) * BATCH_SIZE
            x_batch = x_train[batch_start:batch_end]
            y_batch = y_train_onehot[batch_start:batch_end]
            
            if x_batch.shape[0] == 0: continue # Skip if batch is empty

            step_loss = model_module.learn(x_batch, y_batch)
            epoch_loss_avg.update_state(step_loss)
            if step % 20 == 0:
                print(f"\r Epoch {epoch+1}/{epochs}, Step {step}/{steps_per_epoch}, Avg Loss: {epoch_loss_avg.result():.4f}", end="")
        print(f"\r- Epoch {epoch+1}/{epochs} Complete. Final Avg Loss: {epoch_loss_avg.result():.4f}" + " "*10)
    print(f"initial training complete.")
    return model_module


def run_post_training_and_plot(exact_model_module, approx_model_module,
                               post_train_data_x, post_train_data_y_onehot,
                               post_training_epochs, batch_size):
    """
    Performs post-training, collects per-epoch average loss (normalized to first epoch)
    and average per-epoch training time, and plots them.
    Assumes 'approx_model_module' has an 'approx_learn' method.
    """
    raw_exact_avg_losses_epoch = []
    raw_approx_avg_losses_epoch = []
    exact_avg_time_epoch = []
    approx_avg_time_epoch = []
    
    steps_per_epoch = len(post_train_data_x) // batch_size
    if steps_per_epoch == 0:
        print(f"Error: Not enough post-training data ({len(post_train_data_x)} samples) for batch size {batch_size} to complete one epoch.")
        return
        
    total_steps = post_training_epochs * steps_per_epoch
    print(f"\nStarting post-training phase for {post_training_epochs} epochs ({total_steps} total steps)...")

    if not hasattr(approx_model_module, 'approx_learn'):
        print(f"ERROR: Approximate model does not have an 'approx_learn' method. Cannot proceed with its post-training.")
        run_approx_training = False
    else:
        run_approx_training = True

    for epoch in range(post_training_epochs):
        print(f"Post-Training Epoch {epoch+1}/{post_training_epochs}")
        epoch_exact_loss_metric = tf.keras.metrics.Mean()
        epoch_approx_loss_metric = tf.keras.metrics.Mean()
        
        epoch_exact_total_time = 0.0
        epoch_approx_total_time = 0.0
        actual_steps_in_epoch = 0

        for step in range(steps_per_epoch):
            actual_steps_in_epoch +=1
            batch_start = step * batch_size
            batch_end = (step + 1) * batch_size
            x_batch = post_train_data_x[batch_start:batch_end]
            y_batch = post_train_data_y_onehot[batch_start:batch_end]

            if x_batch.shape[0] == 0: 
                actual_steps_in_epoch -=1 
                continue

            start_time_exact = time.perf_counter()
            exact_loss_val_tensor = exact_model_module.learn(x_batch, y_batch)
            end_time_exact = time.perf_counter()
            epoch_exact_total_time += (end_time_exact - start_time_exact)
            exact_loss_val_np = exact_loss_val_tensor.numpy()
            epoch_exact_loss_metric.update_state(exact_loss_val_np)

            approx_loss_val_np = float('nan') 
            if run_approx_training:
                start_time_approx = time.perf_counter()
                approx_loss_val_tensor = approx_model_module.approx_learn(x_batch, y_batch)
                end_time_approx = time.perf_counter()
                epoch_approx_total_time += (end_time_approx - start_time_approx)
                approx_loss_val_np = approx_loss_val_tensor.numpy()
                epoch_approx_loss_metric.update_state(approx_loss_val_np)
            
            if step % 20 == 0:
                current_approx_loss_display = f"{approx_loss_val_np:.4f}" if not np.isnan(approx_loss_val_np) else "N/A"
                print(f"\r  Step {step}/{steps_per_epoch} | Exact Loss: {exact_loss_val_np:.4f}, Approx Loss: {current_approx_loss_display}", end="")
        
        raw_exact_avg_losses_epoch.append(epoch_exact_loss_metric.result().numpy())
        if actual_steps_in_epoch > 0:
            exact_avg_time_epoch.append(epoch_exact_total_time / actual_steps_in_epoch)
        else:
            exact_avg_time_epoch.append(0.0)

        if run_approx_training and epoch_approx_loss_metric.count.numpy() > 0 :
            raw_approx_avg_losses_epoch.append(epoch_approx_loss_metric.result().numpy())
            if actual_steps_in_epoch > 0:
                 approx_avg_time_epoch.append(epoch_approx_total_time / actual_steps_in_epoch)
            else:
                 approx_avg_time_epoch.append(0.0)
        else: 
            raw_approx_avg_losses_epoch.append(float('nan'))
            approx_avg_time_epoch.append(float('nan'))

        approx_avg_loss_display = f"{epoch_approx_loss_metric.result():.4f}" if epoch_approx_loss_metric.count.numpy() > 0 else "N/A"
        print(f"\rPost-Training Epoch {epoch+1} Avg Losses -> Exact: {epoch_exact_loss_metric.result():.4f}, Approx: {approx_avg_loss_display}" + " "*15)

    # Normalize losses
    first_epoch_exact_loss = raw_exact_avg_losses_epoch[0] if len(raw_exact_avg_losses_epoch) > 0 and raw_exact_avg_losses_epoch[0] != 0 else 1.0
    first_epoch_approx_loss = raw_approx_avg_losses_epoch[0] if len(raw_approx_avg_losses_epoch) > 0 and not np.isnan(raw_approx_avg_losses_epoch[0]) and raw_approx_avg_losses_epoch[0] != 0 else 1.0

    normalized_exact_losses = [loss / first_epoch_exact_loss for loss in raw_exact_avg_losses_epoch]
    normalized_approx_losses = []
    for loss in raw_approx_avg_losses_epoch:
        if not np.isnan(loss):
            normalized_approx_losses.append(loss / first_epoch_approx_loss)
        else:
            normalized_approx_losses.append(float('nan'))
            
    # Plotting
    epochs_axis = np.arange(1, post_training_epochs + 1)
    fig, ax1 = plt.subplots(figsize=(12, 6)) 

    color_exact_loss = 'deepskyblue'
    ax1.set_xlabel('Post-Training Epoch')
    ax1.set_ylabel('Normalized Avg. Loss (Loss / First Epoch Loss)', color='black')
    ax1.plot(epochs_axis, normalized_exact_losses, marker='o', linestyle='-', color=color_exact_loss, label='Exact Model Norm. Loss')
    
    color_approx_loss = 'salmon'
    valid_approx_indices = ~np.isnan(normalized_approx_losses)
    if np.any(valid_approx_indices):
        ax1.plot(epochs_axis[valid_approx_indices], np.array(normalized_approx_losses)[valid_approx_indices], marker='x', linestyle='-', color=color_approx_loss, label='Approx. Model Norm. Loss')
    
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99))
    ax1.grid(True, which="major", ls=":", axis='y', alpha=0.6)
    ax1.set_xticks(epochs_axis) 

    ax2 = ax1.twinx()
    color_exact_time = 'darkblue'
    ax2.set_ylabel('Average Time per Epoch (seconds)', color='dimgray') 
    ax2.plot(epochs_axis, exact_avg_time_epoch, marker='.', linestyle='--', color=color_exact_time, label='Exact Model Avg. Epoch Time')

    color_approx_time = 'darkred'
    valid_approx_time_indices = ~np.isnan(approx_avg_time_epoch) # Re-check for time, could be different if approx training failed early
    if run_approx_training and np.any(valid_approx_time_indices):
        ax2.plot(epochs_axis[valid_approx_time_indices], np.array(approx_avg_time_epoch)[valid_approx_time_indices], marker='+', linestyle='--', color=color_approx_time, label='Approx. Model Avg. Epoch Time')
    
    ax2.tick_params(axis='y', labelcolor='dimgray')
    ax2.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99))

    plt.title('Post-Training: Normalized Avg. Loss and Avg. Time per Epoch')
    fig.tight_layout()
    plt.savefig("plot_post_training_normalized_loss_time_insight2.png")
    plt.show()
    print("Plot for Insight 2 (post-training normalized loss and time) saved as plot_post_training_normalized_loss_time_insight2.png")


def run_insight2_experiment():
    """
    Sets up and runs the experiment for Insight 2: Post-training effects.
    """
    (x_train, y_train, y_train_onehot), (_x_test, _y_test, _y_test_onehot) = load_data()

    # 1. Prepare the Exact Model (partially pre-trained for 1 epoch)
    exact_module_path_insight2 = "mnist_exact_for_posttraining_insight2"
    try:
        exact_model_to_post_train = tf.saved_model.load(exact_module_path_insight2)
        print(f"Loaded pre-trained exact model from {exact_module_path_insight2}")
    except Exception as e:
        print(f"Exact model for Insight 2 not found (Error: {e}). Training for 1 epoch...")
        exact_model_to_post_train = create_mnist_module(BATCH_SIZE)
        exact_model_to_post_train = initial_train_model(exact_model_to_post_train,
                                                        (x_train, y_train, y_train_onehot),
                                                        epochs=1)
        tf.saved_model.save(exact_model_to_post_train, exact_module_path_insight2)
        print(f"Exact model pre-trained for 1 epoch and saved to {exact_module_path_insight2}")

    # 2. Prepare the Approximate Kernel (pre-trained to approximate the above exact_model_to_post_train)
    print("\nCreating and pre-training approximation kernel for Insight 2...")
    approx_kernel_base_structure = get_approx_kernel(FEATURES_SHAPE, OUTPUT_SHAPE, BATCH_SIZE)

    print("Using FuncSubstitute to pre-train the approximate kernel...")
    func_sub_for_approx_pretrain = FuncSubstitute(
        exact_module=exact_model_to_post_train,
        approx_kernel=approx_kernel_base_structure,
        input_shape=FEATURES_SHAPE, 
        batch_size=BATCH_SIZE
    )
    
    if hasattr(func_sub_for_approx_pretrain, 'compile_exact'):
        print("Calling compile_exact() on FuncSubstitute instance for approx pre-training setup...")
        func_sub_for_approx_pretrain.compile_exact()

    num_samples_approx_pretrain = 5000
    epochs_approx_pretrain = 10 
    
    print(f"Pre-training approximation kernel with {num_samples_approx_pretrain} samples for {epochs_approx_pretrain} epochs via FuncSubstitute...")
    func_sub_for_approx_pretrain.train_approx(use_provided=True, 
                                              user_data=x_train[:num_samples_approx_pretrain],
                                              epochs=epochs_approx_pretrain)
    
    approx_model_to_post_train = func_sub_for_approx_pretrain.approx_kernel
    print("Approximation kernel pre-training complete.")

    if not hasattr(approx_model_to_post_train, 'approx_learn'):
        print("ERROR: The pre-trained approximate kernel from FuncSubstitute does not have an 'approx_learn' method.")
        print("       Cannot proceed with post-training comparison for Insight 2 as specified.")
        return

    # 3. Define data and parameters for the post-training phase
    post_train_data_start_idx = num_samples_approx_pretrain 
    post_train_data_count = 10000 
    
    if post_train_data_start_idx + post_train_data_count > len(x_train):
        post_train_data_count = len(x_train) - post_train_data_start_idx
        if post_train_data_count < BATCH_SIZE : 
            post_train_data_start_idx = 0
            post_train_data_count = 10000 
            if post_train_data_count > len(x_train): post_train_data_count = len(x_train)
            print(f"Warning: Adjusted post-training data to start from index 0 with {post_train_data_count} samples.")

    if post_train_data_count < BATCH_SIZE:
        print(f"Error: Insufficient data for post-training phase (need at least {BATCH_SIZE}, have {post_train_data_count}).")
        return
        
    post_train_x_subset = x_train[post_train_data_start_idx : post_train_data_start_idx + post_train_data_count]
    post_train_y_subset_onehot = y_train_onehot[post_train_data_start_idx : post_train_data_start_idx + post_train_data_count]
    
    num_post_training_epochs = 50

    # 4. Run the post-training and plotting
    run_post_training_and_plot(exact_model_module=exact_model_to_post_train,
                               approx_model_module=approx_model_to_post_train,
                               post_train_data_x=post_train_x_subset,
                               post_train_data_y_onehot=post_train_y_subset_onehot,
                               post_training_epochs=num_post_training_epochs,
                               batch_size=BATCH_SIZE)

    

if __name__ == "__main__":
    run_insight2_experiment()