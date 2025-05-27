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
            'max_length': 128,
            'task': 'masked_lm'
        },
        'gpt2': {
            'model_class': FlaxGPT2LMHeadModel,
            'tokenizer_class': GPT2Tokenizer,
            'pretrained': 'gpt2',  # You can also use 'distilgpt2' for smaller size
            'max_length': 128,
            'task': 'generation'
        },
        'llama': {
            'model_class': FlaxLlamaForCausalLM,
            'tokenizer_class': LlamaTokenizer,
            'pretrained': 'openlm-research/open_llama_3b',  # Small Llama model
            'max_length': 128,
            'task': 'generation'
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
        self.model = self.config['model_class'].from_pretrained(self.config['pretrained'])
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
        
        # Load compiled module
        self.compiled_module = ToolBox.load_mlir_from_file(mlir_path).jit_forward_fn
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
            print("Enter text to see next token predictions.")
            print("Example: 'The weather today is'")
        
        print("\nType 'quit' to exit.\n")
        
        while True:
            text = input("> ")
            if text.lower() == 'quit':
                break
            
            if not text.strip():
                continue
            
            try:
                result = self.run_inference(text)
                self._print_results(result)
            except Exception as e:
                print(f"Error: {e}")
    
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
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for compilation')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Maximum sequence length')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save compiled MLIR files')
    
    args = parser.parse_args()
    
    # Initialize model compiler/runner
    runner = ModelCompilerRunner(args.model, args.max_length)
    
    # Compile model
    runner.compile_model(args.batch_size, args.save_path)
    
    if not args.compile_only:
        if args.text:
            # Non-interactive mode
            result = runner.run_inference(args.text)
            runner._print_results(result)
        else:
            # Interactive mode
            runner.interactive_demo()


if __name__ == "__main__":
    main()