
SRC_FILE=$1

clang-15 -S $1.ll --target=riscv64 -march=rv64gc

