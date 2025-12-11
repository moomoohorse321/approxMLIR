// RUN: approx-opt %s | approx-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = approx.foo %{{.*}} : i32
        %res = approx.foo %0 : i32
        return
    }
}
