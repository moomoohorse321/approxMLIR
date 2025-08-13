module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.mlir.global internal constant @str0("%f \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  "approxMLIR.util.annotation.decision_tree"() <{
    func_name = "accumulate_point_to_centroid",
    transform_type = "loop_perforate",
    num_thresholds = 1 : i32,
    thresholds_uppers = array<i32: 100>,
    thresholds_lowers = array<i32: 2>,
    decision_values = array<i32: 0, 1>,
    thresholds = array<i32: 50>,
    decisions = array<i32: 1, 2>
}> : () -> ()
  func.func @accumulate_point_to_centroid(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg2 : i32 to index
    scf.for %arg3 = %c0 to %0 step %c1 {
      %1 = memref.load %arg0[%arg3] : memref<?xf64>
      %2 = memref.load %arg1[%arg3] : memref<?xf64>
      %3 = arith.addf %2, %1 : f64
      memref.store %3, %arg1[%arg3] : memref<?xf64>
    }
    return
  }
  func.func @base(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
    call @accumulate_point_to_centroid(%arg1, %arg2, %arg0) : (memref<?xf64>, memref<?xf64>, i32) -> ()
    return
  }
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst = arith.constant 0.000000e+00 : f64
    %cst_0 = arith.constant 5.000000e+00 : f64
    %cst_1 = arith.constant 4.000000e+00 : f64
    %cst_2 = arith.constant 3.000000e+00 : f64
    %cst_3 = arith.constant 2.000000e+00 : f64
    %cst_4 = arith.constant 1.000000e+00 : f64
    %alloca = memref.alloca() : memref<5xf64>
    %alloca_5 = memref.alloca() : memref<5xf64>
    affine.store %cst_4, %alloca_5[0] : memref<5xf64>
    affine.store %cst_3, %alloca_5[1] : memref<5xf64>
    affine.store %cst_2, %alloca_5[2] : memref<5xf64>
    affine.store %cst_1, %alloca_5[3] : memref<5xf64>
    affine.store %cst_0, %alloca_5[4] : memref<5xf64>
    affine.store %cst, %alloca[0] : memref<5xf64>
    affine.store %cst, %alloca[1] : memref<5xf64>
    affine.store %cst, %alloca[2] : memref<5xf64>
    affine.store %cst, %alloca[3] : memref<5xf64>
    affine.store %cst, %alloca[4] : memref<5xf64>
    %cast = memref.cast %alloca_5 : memref<5xf64> to memref<?xf64>
    %cast_6 = memref.cast %alloca : memref<5xf64> to memref<?xf64>
    call @base(%c5_i32, %cast, %cast_6, %c2_i32) : (i32, memref<?xf64>, memref<?xf64>, i32) -> ()
    scf.for %arg0 = %c0 to %c5 step %c1 {
      %0 = llvm.mlir.addressof @str0 : !llvm.ptr
      %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
      %2 = memref.load %alloca[%arg0] : memref<5xf64>
      %3 = llvm.call @printf(%1, %2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
    }
    return %c0_i32 : i32
  }
}
