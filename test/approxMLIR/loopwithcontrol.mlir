// RUN: approxMLIR-opt %s > %S/%basename.mlir

module {


  func.func @decision_tree(%state: f32, %num_thresholds: i32, 
                         %thresholds_uppers: tensor<3xf32>,
                         %thresholds_lowers: tensor<3xf32>,
                         %decision_values: tensor<3xi32>,
                         %thresholds: tensor<3xf32>,
                         %decisions: tensor<3xi32>) {
    // Implementation
    return
  }

  func.func @knob_start() {
    return 
  }

  func.func @knob_end() {
    return 
  }

  func.func @main(%11 : index) {
    %12 = memref.alloc(%11) : memref<?xf32>
    %28 = memref.alloca() : memref<16x16xf32>
    
    // Loop through each row of the 2D array
    affine.for %i = 0 to 16 {
      // Initialize max value with the first element of the row
      %init_val = affine.load %28[%i, 0] : memref<16x16xf32>
      %max_val = memref.alloca() : memref<f32>
      affine.store %init_val, %max_val[] : memref<f32>  // No indices for 0D memref
      
      %i_i32 = arith.index_cast %i : index to i32
      %i_f32 = arith.sitofp %i_i32 : i32 to f32
      
      // Find max in the current row

      // todo for the emitting pass, 
      func.call @knob_start() : () -> ()

      %num_thresholds = arith.constant 3 : i32
      %thresholds_uppers = arith.constant dense<[2.0, 3.0, 4.0]> : tensor<3xf32>
      %thresholds_lowers = arith.constant dense<[0.0, 2.0, 3.0]> : tensor<3xf32>
      %decision_values = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
      %thresholds = arith.constant dense<[0.0, 0.0, 0.0]> : tensor<3xf32>
      %decisions = arith.constant dense<[0, 0, 0]> : tensor<3xi32>

      func.call @decision_tree(%i_f32, %num_thresholds, %thresholds_uppers, %thresholds_lowers, %decision_values, %thresholds, %decisions) : (f32, i32, tensor<3xf32>, tensor<3xf32>, tensor<3xi32>, tensor<3xf32>, tensor<3xi32>) -> ()
    
      affine.for %j = 1 to 16 {
        // here inject the observeOp, decideOp (optionally checkOp)
        %curr_val = affine.load %28[%i, %j] : memref<16x16xf32>
        %prev_max = affine.load %max_val[] : memref<f32>  // No indices for 0D memref
        
        // Compare current value with current max
        %cmp = arith.cmpf ogt, %curr_val, %prev_max : f32
        %new_max = arith.select %cmp, %curr_val, %prev_max : f32
        
        // Update max value
        affine.store %new_max, %max_val[] : memref<f32>  // No indices for 0D memref
      }

      func.call @knob_end() : () -> ()
      
      // Store the max value to the dynamic array
      %final_max = affine.load %max_val[] : memref<f32>  // No indices for 0D memref
      affine.store %final_max, %12[%i] : memref<?xf32>
    }
    return 
  }

  func.func @func1(%11 : index, %c : i1) {
    %12 = memref.alloc(%11) : memref<?xf32>
      %28 = memref.alloca() : memref<16x16xf32>
      affine.for %arg4 = 0 to 16 {
        affine.for %barg4 = 0 to 16 {
          %29 = scf.if %c -> f32 {
            %l1 = affine.load %28[%arg4, 0] : memref<16x16xf32>
            scf.yield %l1 : f32
          } else {
            %a = affine.load %12[1] : memref<?xf32>
            affine.store %a, %12[0] : memref<?xf32>
            scf.yield %a : f32
          }
          %a = affine.load %12[0] : memref<?xf32>
          %z = arith.addf %a, %29 : f32
          affine.store %z, %12[0] : memref<?xf32>
        }
    }
    return
  }
}
