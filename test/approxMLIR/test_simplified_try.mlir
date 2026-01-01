// ============================================================================
// Test: Simplified TryOp Transformation Pipeline
// ============================================================================
// This file demonstrates the simplified safety contract transformation.
//
// Key simplification: No "candidates" concept - the yield's operands are
// implicitly the candidates to validate.

// ============================================================================
// STAGE 0: Input with Annotation
// ============================================================================
// The annotation specifies: function, checker, and recovery function names

module {
  // Runtime functions (declared externally or defined elsewhere)
  func.func private @recover_impl(%arg0: f32, %arg1: f32) -> f32
  func.func private @checker_impl(%result: f32, %arg0: f32, %arg1: f32) -> i1

  // Safety contract annotation
  "approxMLIR.util.annotation.try"() <{
    func_name = "_sum",
    recover = "@recover_impl",
    checker = "@checker_impl"
  }> : () -> ()

  // Function with existing KnobOp (created by emit-approx pass)
  func.func @_sum(%sum_iter: f32, %t: f32) -> f32 {
    %result = "approxMLIR.knob"(%sum_iter, %sum_iter, %t) <{
      id = 0 : i32,
      rf = 0 : i32,
      QoS_in = array<i32:>,
      QoS_out = array<i32:>,
      transform_type = "try_check_recover"
    }> ({
      // Approximate computation
      %attempt = arith.addf %sum_iter, %t : f32
      
      // Original yield (TryOp will be injected BEFORE this)
      "approxMLIR.yield"(%attempt) : (f32) -> ()
    }) : (f32, f32, f32) -> f32
    
    return %result : f32
  }
}

// ============================================================================
// STAGE 1: After emit-safety (EmitTryFromAnnotation pattern)
// ============================================================================
// The annotation is processed and TryOp is injected before the yield.
// RUN: approx-opt --emit-safety %s | FileCheck %s --check-prefix=EMIT

// EMIT: module {
// EMIT:   func.func private @recover_impl(%arg0: f32, %arg1: f32) -> f32
// EMIT:   func.func private @checker_impl(%result: f32, %arg0: f32, %arg1: f32) -> i1
// EMIT:
// EMIT:   // Annotation is GONE
// EMIT:
// EMIT:   func.func @_sum(%sum_iter: f32, %t: f32) -> f32 {
// EMIT:     %result = "approxMLIR.knob"(%sum_iter, %sum_iter, %t) <{...}> ({
// EMIT:       %attempt = arith.addf %sum_iter, %t : f32
// EMIT:
// EMIT:       // NEW: TryOp injected before yield
// EMIT:       "approxMLIR.try"(%sum_iter, %t) <{recover = "@recover_impl"}> ({
// EMIT:         ^bb0(%arg0: f32, %arg1: f32):
// EMIT:           // Checker called with: yield_operands + recovery_args
// EMIT:           %valid = func.call @checker_impl(%attempt, %arg0, %arg1)
// EMIT:               : (f32, f32, f32) -> i1
// EMIT:           "approxMLIR.yield"(%valid) : (i1) -> ()
// EMIT:       }) : (f32, f32) -> ()
// EMIT:
// EMIT:       // Original yield unchanged at this stage
// EMIT:       "approxMLIR.yield"(%attempt) : (f32) -> ()
// EMIT:     }) : (f32, f32, f32) -> f32
// EMIT:     return %result : f32
// EMIT:   }
// EMIT: }

// ============================================================================
// STAGE 2: After emit-safety (LowerTryOpPattern - same pass, later iteration)
// ============================================================================
// TryOp is lowered to scf.if and the yield is updated.
// The emit-safety pass runs both patterns iteratively.

// LOWER: module {
// LOWER:   func.func private @recover_impl(%arg0: f32, %arg1: f32) -> f32
// LOWER:   func.func private @checker_impl(%result: f32, %arg0: f32, %arg1: f32) -> i1
// LOWER:
// LOWER:   func.func @_sum(%sum_iter: f32, %t: f32) -> f32 {
// LOWER:     %result = "approxMLIR.knob"(%sum_iter, %sum_iter, %t) <{...}> ({
// LOWER:       
// LOWER:       // Original computation
// LOWER:       %attempt = arith.addf %sum_iter, %t : f32
// LOWER:
// LOWER:       // Check region INLINED
// LOWER:       %valid = func.call @checker_impl(%attempt, %sum_iter, %t)
// LOWER:           : (f32, f32, f32) -> i1
// LOWER:
// LOWER:       // scf.if created from TryOp
// LOWER:       %if_result = scf.if %valid -> (f32) {
// LOWER:         // Check passed: use original yield value
// LOWER:         scf.yield %attempt : f32
// LOWER:       } else {
// LOWER:         // Check failed: call recovery
// LOWER:         %recovered = func.call @recover_impl(%sum_iter, %t) 
// LOWER:             : (f32, f32) -> f32
// LOWER:         scf.yield %recovered : f32
// LOWER:       }
// LOWER:
// LOWER:       // Yield UPDATED to use if result
// LOWER:       "approxMLIR.yield"(%if_result) : (f32) -> ()
// LOWER:
// LOWER:     }) : (f32, f32, f32) -> f32
// LOWER:     return %result : f32
// LOWER:   }
// LOWER: }

// ============================================================================
// Full example with for loop (matches original gold intermediate)
// ============================================================================

module {
  func.func private @recover_impl(%sum_iter: f32, %t: f32) -> f32
  func.func private @checker_impl(%candidate: f32, %sum_iter: f32, %t: f32) -> i1

  // Annotation
  "approxMLIR.util.annotation.try"() <{
    func_name = "_sum",
    recover = "@recover_impl", 
    checker = "@checker_impl"
  }> : () -> ()

  func.func @_sum(%sum_iter: f32, %t: f32) -> (f32) {
    %result = "approxMLIR.knob"(%sum_iter, %sum_iter, %t) <{
      id = 0 : i32, rf = 0 : i32,
      QoS_in = array<i32:>, QoS_out = array<i32:>,
      transform_type = "try_check_recover"
    }> ({
      %sum_next = arith.addf %sum_iter, %t : f32
      "approxMLIR.yield"(%sum_next) : (f32) -> ()
    }) : (f32, f32, f32) -> f32
    return %result : f32
  }

  func.func @for_with_yield(%buffer: memref<1024xf32>) -> (f32) {
    %sum_0 = arith.constant 0.0 : f32
    %sum = affine.for %i = 0 to 10 step 2 iter_args(%sum_iter = %sum_0) -> (f32) {
      %t = affine.load %buffer[%i] : memref<1024xf32>
      %sum_next = func.call @_sum(%sum_iter, %t) : (f32, f32) -> f32
      affine.yield %sum_next : f32
    }
    return %sum : f32
  }
}

// ============================================================================
// After full pipeline (emit-safety):
// ============================================================================

// FULL: func.func @_sum(%sum_iter: f32, %t: f32) -> f32 {
// FULL:   %result = "approxMLIR.knob"(...) ({
// FULL:     %sum_next = arith.addf %sum_iter, %t : f32
// FULL:     %valid = func.call @checker_impl(%sum_next, %sum_iter, %t) -> i1
// FULL:     %final = scf.if %valid -> (f32) {
// FULL:       scf.yield %sum_next : f32
// FULL:     } else {
// FULL:       %rec = func.call @recover_impl(%sum_iter, %t) -> f32
// FULL:       scf.yield %rec : f32
// FULL:     }
// FULL:     "approxMLIR.yield"(%final) : (f32) -> ()
// FULL:   }) : (...) -> f32
// FULL:   return %result : f32
// FULL: }

// ============================================================================
// Design Notes
// ============================================================================
//
// Q: Why place TryOp before yield instead of having it produce results?
// A: Simplicity. The yield's operands naturally represent "what we computed".
//    The TryOp validates those values. No need to thread results through.
//
// Q: How does the checker know what to validate?
// A: The check region can reference any SSA value defined earlier in the knob.
//    The emit pass builds the checker call with yield_operands + recovery_args.
//
// Q: What if there are multiple yields?
// A: Currently assumes single-block knobs with one yield. Extension for
//    multi-block knobs would need to handle each yield individually.
//
// Q: Can TryOp be nested or appear outside KnobOp?
// A: Not in this simplified design. TryOp semantics depend on having a
//    following yield to define the "success" values.
