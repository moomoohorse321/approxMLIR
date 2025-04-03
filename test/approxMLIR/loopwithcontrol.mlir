// RUN: approxMLIR-opt %s > %S/%basename.mlir

module {

  func.func @get_state(%x : i32) -> (f32) {
    %a = arith.constant 4 : i32
    %c = arith.cmpi slt, %x, %a : i32
    %state = scf.if %c -> f32 {
        %state1 = arith.constant 1.0 : f32
        scf.yield %state1 : f32
    } else {
        %state2 = arith.constant 2.0 : f32
        scf.yield %state2 : f32
    }
    return %state : f32
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
      
      // Find max in the current row
      func.call @knob_start() : () -> ()
      %i_i32 = arith.index_cast %i : index to i32
      %6 = func.call @get_state(%i_i32) : (i32) -> (f32)
      
      affine.for %j = 1 to 16 {
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
