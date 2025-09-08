from approxMLIR import ToolBox
import jax
import jax.numpy as jnp
from jax import export
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir
from transformers import (
    FlaxBertForMaskedLM, 
    FlaxGPT2LMHeadModel,
    FlaxLlamaForCausalLM,
    BertTokenizer,
    GPT2Tokenizer,
    LlamaTokenizer,
    AutoTokenizer
)
import numpy as np
import argparse
import os

# Disable logging for cleaner output
import logging
logging.disable(logging.WARNING)

class ModelCompilerRunner:
    """A class to compile and run various transformer models with IREE."""
    
    # Model configurations
    MODEL_CONFIGS = {
        'bert': {
            'model_class': FlaxBertForMaskedLM,
            'tokenizer_class': BertTokenizer,
            'pretrained': 'bert-base-uncased',
            'max_length': 512,  # BERT's actual max length
            'task': 'masked_lm',
            'from_pt': False
        },
        'gpt2': {
            'model_class': FlaxGPT2LMHeadModel,
            'tokenizer_class': GPT2Tokenizer,
            'pretrained': 'gpt2',  # You can also use 'distilgpt2' for smaller size
            'max_length': 1024,  # GPT-2's actual max length
            'task': 'generation',
            'from_pt': False
        },
        'llama': {
            'model_class': FlaxLlamaForCausalLM,
            'tokenizer_class': LlamaTokenizer,
            'pretrained': 'openlm-research/open_llama_3b',  # Small Llama model
            'max_length': 2048,  # Llama's actual max length
            'task': 'generation',
            'from_pt': True
        }
    }
    
    def __init__(self, model_type='bert', max_length=None):
        """Initialize the compiler/runner with a specific model type."""
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Model type '{model_type}' not supported. Choose from: {list(self.MODEL_CONFIGS.keys())}")
        
        self.model_type = model_type
        self.config = self.MODEL_CONFIGS[model_type]
        
        # Override max_length if provided
        if max_length:
            self.config['max_length'] = max_length
        
        # Load model and tokenizer
        print(f"Loading {model_type} model: {self.config['pretrained']}...")
        self.model = self.config['model_class'].from_pretrained(self.config['pretrained'], from_pt = self.config['from_pt'])
        self.tokenizer = self.config['tokenizer_class'].from_pretrained(self.config['pretrained'])
        
        # Set pad token if not exists (needed for GPT2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.compiled_module = None
        
    def get_stablehlo_asm(self, module_str):
        """Returns prettyprint of StableHLO module without large constants."""
        with jax_mlir.make_ir_context():
            stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
            return stablehlo_module.operation.get_asm(large_elements_limit=20)
    
    def compile_model(self, batch_size=1, save_path=None):
        """Compile the model to StableHLO and optionally save to file."""
        print(f"Compiling {self.model_type} model...")
        
        # Create example input shape
        sample_shape = (batch_size, self.config['max_length'])
        sample_input = jnp.ones(sample_shape, dtype=jnp.int32)
        
        # Define forward function based on task type
        if self.config['task'] == 'masked_lm':
            def forward_fn(input_ids):
                return self.model(input_ids).logits
        else:  # generation tasks
            def forward_fn(input_ids):
                return self.model(input_ids).logits
        
        # JIT compile and export
        jitted_fn = jax.jit(forward_fn)
        input_shape = jax.ShapeDtypeStruct(sample_input.shape, sample_input.dtype)
        exported = export.export(jitted_fn)(input_shape)
        stablehlo_module = exported.mlir_module()
        
        # Save MLIR files
        if save_path is None:
            save_path = f"{self.model_type}_model"
        
        # Save raw MLIR
        mlir_path = f"{save_path}.mlir"
        with open(mlir_path, "w") as f:
            f.write(stablehlo_module)
        print(f"Saved raw MLIR to {mlir_path}")
        
        # Save pretty-printed version
        pretty_mlir_path = f"{save_path}_pretty.mlir"
        with open(pretty_mlir_path, "w") as f:
            f.write(self.get_stablehlo_asm(stablehlo_module))
        print(f"Saved pretty MLIR to {pretty_mlir_path}")
        
        # optimize mlir
        
        # Load compiled module
        self.compiled_module = ToolBox.load_mlir_from_file(mlir_path, backend_name = "gpu").jit_forward_fn
        print("Model compiled successfully!")
        
        return mlir_path
    
    def tokenize_text(self, text, return_tensors='jax'):
        """Tokenize input text with proper padding and truncation."""
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors='np'  # Get numpy first, then convert
        )
        
        if return_tensors == 'jax':
            # Convert to JAX arrays
            input_ids = jnp.array(encoded['input_ids'])
            return input_ids
        return encoded['input_ids']
    
    def run_inference(self, text_input, decode_output=True):
        """Run inference on text input and optionally decode the output."""
        if not self.compiled_module:
            raise RuntimeError("Model not compiled yet. Run compile_model() first.")
        
        print(f"\nRunning inference on: '{text_input}'")
        
        # Tokenize input
        input_ids = self.tokenize_text(text_input)
        print(f"Tokenized shape: {input_ids.shape}")
        
        # Run inference
        output_logits = self.compiled_module.main(input_ids).to_host()
        print(f"Output shape: {output_logits.shape}")
        
        if decode_output:
            # Different decoding strategies based on task
            if self.config['task'] == 'masked_lm':
                # For BERT: show predictions for [MASK] tokens
                return self._decode_masked_lm(text_input, input_ids, output_logits)
            else:
                # For GPT2/Llama: show next token predictions
                return self._decode_generation(input_ids, output_logits)
        
        return output_logits
    
    def _decode_masked_lm(self, original_text, input_ids, logits):
        """Decode predictions for masked language modeling."""
        # Find [MASK] token positions
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = np.where(input_ids == mask_token_id)[1]
        
        if len(mask_positions) == 0:
            print("No [MASK] tokens found in input!")
            return logits
        
        results = []
        for pos in mask_positions:
            # Get top 5 predictions for this position
            top_5_tokens = np.argsort(logits[0, pos])[-5:][::-1]
            top_5_probs = jax.nn.softmax(logits[0, pos])[top_5_tokens]
            
            predictions = []
            for token_id, prob in zip(top_5_tokens, top_5_probs):
                token = self.tokenizer.decode([token_id])
                predictions.append(f"{token} ({prob:.3f})")
            
            results.append({
                'position': int(pos),
                'predictions': predictions
            })
        
        return {
            'original_text': original_text,
            'mask_predictions': results
        }
    
    def _decode_generation(self, input_ids, logits):
        """Decode predictions for text generation."""
        # Get the last token's predictions
        last_logits = logits[0, -1, :]
        
        # Get top 5 next token predictions
        top_5_tokens = np.argsort(last_logits)[-5:][::-1]
        top_5_probs = jax.nn.softmax(last_logits)[top_5_tokens]
        
        predictions = []
        for token_id, prob in zip(top_5_tokens, top_5_probs):
            token = self.tokenizer.decode([token_id])
            predictions.append(f"{token} ({prob:.3f})")
        
        # Decode the input text
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        return {
            'input_text': input_text,
            'next_token_predictions': predictions
        }
    
    def generate_text(self, prompt, num_tokens=50, temperature=1.0, stop_at_sentence=False):
        """Generate text iteratively, one token at a time.
        
        Args:
            prompt: Initial text prompt
            num_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            stop_at_sentence: Stop at sentence-ending punctuation
        
        Returns:
            Dictionary with generated text and token-by-token info
        """
        if not self.compiled_module:
            raise RuntimeError("Model not compiled yet. Run compile_model() first.")
        
        if self.config['task'] == 'masked_lm':
            print("Text generation is only supported for GPT2 and Llama models.")
            return None
        
        print(f"\nGenerating text from prompt: '{prompt}'")
        print(f"Parameters: max_tokens={num_tokens}, temperature={temperature}, stop_at_sentence={stop_at_sentence}")
        print("-" * 50)
        
        # Tokenize the prompt with padding
        encoded = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.config['max_length'],
            return_tensors='np'
        )
        
        input_ids = encoded['input_ids'][0]  # Get first batch element
        attention_mask = encoded['attention_mask'][0]
        
        # Find the first padding position (where we'll start generating)
        pad_token_id = self.tokenizer.pad_token_id
        pad_positions = np.where(input_ids == pad_token_id)[0]
        
        if len(pad_positions) == 0:
            print("Warning: No padding tokens available. Input is at maximum length.")
            return {
                'generated_text': prompt,
                'tokens_generated': 0,
                'reason': 'max_length_reached'
            }
        
        # Track generation
        generated_tokens = []
        generation_position = pad_positions[0]  # First padding position
        stop_tokens = {self.tokenizer.convert_tokens_to_ids(t) for t in ['.', '!', '?'] if t in self.tokenizer.get_vocab()}
        
        # Generate tokens one by one
        for i in range(min(num_tokens, len(pad_positions))):
            # Prepare input as JAX array
            jax_input = jnp.array([input_ids])  # Add batch dimension
            
            # Run model
            logits = self.compiled_module.main(jax_input).to_host()
            
            # Get logits for the last non-pad token
            current_position = generation_position + i - 1
            next_token_logits = logits[0, current_position, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Sample from the distribution
            probs = jax.nn.softmax(next_token_logits)
            next_token_id = np.random.choice(len(probs), p=np.array(probs))
            
            # Update input_ids
            input_ids[generation_position + i] = next_token_id
            attention_mask[generation_position + i] = 1
            
            # Decode the token
            token_text = self.tokenizer.decode([next_token_id])
            generated_tokens.append({
                'token_id': int(next_token_id),
                'token_text': token_text,
                'position': int(generation_position + i)
            })
            
            # Print token as it's generated
            print(token_text, end='', flush=True)
            
            # Check stopping conditions
            if stop_at_sentence and next_token_id in stop_tokens:
                print(f"\n[Stopped at sentence end]")
                break
            
            if next_token_id == self.tokenizer.eos_token_id:
                print(f"\n[Reached end-of-sequence token]")
                break
        
        print("\n" + "-" * 50)
        
        # Decode the full generated text
        generated_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'tokens_generated': len(generated_tokens),
            'token_details': generated_tokens,
            'reason': 'completed'
        }
    
    def test_generation_capabilities(self):
        """Test the model's generation capabilities with various prompts."""
        if self.config['task'] != 'generation':
            print(f"Generation test is only available for GPT2 and Llama models, not {self.model_type}")
            return
        
        if not self.compiled_module:
            print("Compiling model first...")
            self.compile_model()
        
        test_prompts = [
            {
                'prompt': "The future of artificial intelligence is",
                'num_tokens': 30,
                'temperature': 0.8
            },
            {
                'prompt': "Once upon a time in a distant galaxy",
                'num_tokens': 40,
                'temperature': 1.0
            },
            {
                'prompt': "The most important scientific discovery",
                'num_tokens': 25,
                'temperature': 0.7,
                'stop_at_sentence': True
            },
            {
                'prompt': "In the year 2050",
                'num_tokens': 35,
                'temperature': 0.9
            }
        ]
        
        print(f"\n{'='*60}")
        print(f"Testing {self.model_type.upper()} Generation Capabilities")
        print(f"{'='*60}\n")
        
        for i, test in enumerate(test_prompts, 1):
            print(f"Test {i}/{len(test_prompts)}")
            result = self.generate_text(**test)
            print(f"\nFull generated text:\n{result['generated_text']}\n")
            print(f"Tokens generated: {result['tokens_generated']}")
            print("="*60 + "\n")
            
    def interactive_demo(self):
        """Run an interactive demo where users can input text."""
        if not self.compiled_module:
            print("Compiling model first...")
            self.compile_model()
        
        print(f"\n{'='*50}")
        print(f"Interactive {self.model_type.upper()} Demo")
        print(f"{'='*50}")
        
        if self.config['task'] == 'masked_lm':
            print("Enter text with [MASK] tokens to see predictions.")
            print("Example: 'The capital of France is [MASK].'")
        else:
            print("Enter text to generate completions.")
            print("Commands:")
            print("  - Just enter text for single token prediction")
            print("  - 'gen: <text>' for multi-token generation")
            print("  - 'gen: <text> | <num_tokens>' to specify token count")
            print("  - 'test' to run generation tests")
            print("\nExamples:")
            print("  - gen: Once upon a time")
            print("  - gen: The weather today | 20")
        
        print("\nType 'quit' to exit.\n")
        
        while True:
            text = input("> ")
            if text.lower() == 'quit':
                break
            
            if not text.strip():
                continue
            
            try:
                if text.lower() == 'test' and self.config['task'] == 'generation':
                    self.test_generation_capabilities()
                elif text.startswith('gen:') and self.config['task'] == 'generation':
                    # Parse generation command
                    parts = text[4:].strip().split('|')
                    prompt = parts[0].strip()
                    num_tokens = 30  # default
                    
                    if len(parts) > 1:
                        try:
                            num_tokens = int(parts[1].strip())
                        except ValueError:
                            print("Invalid token count, using default 30")
                    
                    result = self.generate_text(prompt, num_tokens=num_tokens)
                    print(f"\nGenerated {result['tokens_generated']} tokens")
                else:
                    # Single token prediction
                    result = self.run_inference(text)
                    self._print_results(result)
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    def _print_results(self, result):
        """Pretty print the results."""
        print("\nResults:")
        print("-" * 30)
        
        if 'mask_predictions' in result:
            # BERT results
            print(f"Original: {result['original_text']}")
            for pred in result['mask_predictions']:
                print(f"\n[MASK] at position {pred['position']}:")
                for i, p in enumerate(pred['predictions'], 1):
                    print(f"  {i}. {p}")
        else:
            # Generation results
            print(f"Input: {result['input_text']}")
            print(f"\nNext token predictions:")
            for i, p in enumerate(result['next_token_predictions'], 1):
                print(f"  {i}. {p}")
        
        print("-" * 30)


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Compile and run transformer models with IREE')
    parser.add_argument('--model', type=str, default='bert', 
                       choices=['bert', 'gpt2', 'llama'],
                       help='Model type to use')
    parser.add_argument('--compile-only', action='store_true',
                       help='Only compile the model without running inference')
    parser.add_argument('--text', type=str, 
                       help='Text to run inference on (non-interactive mode)')
    parser.add_argument('--generate', type=str,
                       help='Text prompt for multi-token generation (GPT2/Llama only)')
    parser.add_argument('--num-tokens', type=int, default=30,
                       help='Number of tokens to generate (with --generate)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature for generation')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for compilation')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Maximum sequence length (default: model-specific)')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save compiled MLIR files')
    parser.add_argument('--test-generation', action='store_true',
                       help='Run generation capability tests (GPT2/Llama only)')
    
    args = parser.parse_args()
    
    # Initialize model compiler/runner
    runner = ModelCompilerRunner(args.model, args.max_length)
    
    # Compile model
    runner.compile_model(args.batch_size, args.save_path)
    
    if not args.compile_only:
        if args.test_generation:
            # Run generation tests
            runner.test_generation_capabilities()
        elif args.generate:
            # Multi-token generation mode
            result = runner.generate_text(
                args.generate, 
                num_tokens=args.num_tokens,
                temperature=args.temperature
            )
            print(f"\nGenerated {result['tokens_generated']} tokens")
        elif args.text:
            # Single token prediction mode
            result = runner.run_inference(args.text)
            runner._print_results(result)
        else:
            # Interactive mode
            runner.interactive_demo()


if __name__ == "__main__":
    main()