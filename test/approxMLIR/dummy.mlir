// RUN: approxMLIR-opt %s | approxMLIR-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = approxMLIR.foo %{{.*}} : i32
        %res = approxMLIR.foo %0 : i32
        return
    }
}
