target triple = "riscv64-unknown-unknown"

define i32 @main() {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  ret i32 2
}
