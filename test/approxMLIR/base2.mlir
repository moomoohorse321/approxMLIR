module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  
  "approxMLIR.util.annotation.decision_tree"() <{decision_values = array<i32: 0, 1>, decisions = array<i32: 0, 1>, func_name = "base", num_thresholds = 1 : i32, thresholds = array<i32: 2>, thresholds_lowers = array<i32: 0>, thresholds_uppers = array<i32: 4>, transform_type = "func_substitute"}> : () -> ()

  "approxMLIR.util.annoatation.convert_to_call"() <{func_name = "base"}> : () -> ()

  llvm.mlir.global internal constant @str0("Base function result: %d\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func @approx_base(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = scf.for %arg2 = %c1 to %0 step %c2 iter_args(%arg3 = %c0_i32) -> (i32) {
      %2 = arith.index_cast %arg2 : index to i32
      %3 = arith.muli %2, %arg1 : i32
      %4 = arith.addi %arg3, %3 : i32
      scf.yield %4 : i32
    }
    return %1 : i32
  }
  func.func @base(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = scf.for %arg2 = %c1 to %0 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
      %2 = arith.index_cast %arg2 : index to i32
      %3 = arith.muli %2, %arg1 : i32
      %4 = arith.addi %arg3, %3 : i32
      scf.yield %4 : i32
    }
    return %1 : i32
  }
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c5_i32 = arith.constant 5 : i32
    %0 = llvm.mlir.addressof @str0 : !llvm.ptr
    %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<26 x i8>
    %2 = call @base(%c5_i32, %c2_i32) : (i32, i32) -> i32
    %3 = llvm.call @printf(%1, %2) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    return %c0_i32 : i32
  }
}
