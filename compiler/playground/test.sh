#!/bin/bash

assert() {
    expected="$(( ($1 % 256 + 256) % 256 ))"
    input="$2"

    ./build/ganymede "$input" > ./build/tmp.s || exit
    # ./build/ganymede "-s" "$input" -o ./build/tmp.s || exit

    # riscv64-linux-gnu-gcc -static -o ./build/tmp ./build/tmp.s
    riscv64-linux-gnu-gcc -static -o ./build/tmp ./build/tmp.s
    qemu-riscv64-static ./build/tmp

    actual="$?"

    if [ "$actual" = "$expected" ]; then
        echo "$input => $actual" | paste -s -d ' '
    else
        echo "$input => $expected expected, but got $actual"
        exit 1
    fi
}

assert 3 "int main() { int a = 23; if (a > 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 10; if (a > 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 23; if (a < 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 10; if (a < 10) { return 3; } return 0; }";

assert 0 "int main() { int a = 23; if (a <= 10) { return 3; } return 0; }";
assert 3 "int main() { int a = 10; if (a <= 10) { return 3; } return 0; }";
assert 3 "int main() { int a = 9; if (a <= 10) { return 3; } return 0; }";

assert 3 "int main() { int a = 23; if (a >= 10) { return 3; } return 0; }";
assert 3 "int main() { int a = 10; if (a >= 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 9; if (a >= 10) { return 3; } return 0; }";

assert 0 "int main() { int a = 23; if (a == 10) { return 3; } return 0; }";
assert 3 "int main() { int a = 10; if (a == 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 9; if (a == 10) { return 3; } return 0; }";

assert 3 "int main() { int a = 23; if (a != 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 10; if (a != 10) { return 3; } return 0; }";
assert 3 "int main() { int a = 9; if (a != 10) { return 3; } return 0; }";

assert 3 "int main() { int a = 9; if (a || 10) { return 3; } return 0; }";
assert 3 "int main() { int a = 0; if (a || 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 0; if (a || 0) { return 3; } return 0; }";
assert 3 "int main() { int a = 23; if (a || 0) { return 3; } return 0; }";

assert 3 "int main() { int a = 9; if (a && 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 0; if (a && 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 0; if (a && 0) { return 3; } return 0; }";
assert 0 "int main() { int a = 23; if (a && 0) { return 3; } return 0; }";

assert 3 "int main() { int a = 9; if (a | 10) { return 3; } return 0; }";
assert 3 "int main() { int a = 0; if (a | 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 0; if (a | 0) { return 3; } return 0; }";
assert 3 "int main() { int a = 9; if (a | 0) { return 3; } return 0; }";

assert 3 "int main() { int a = 9; if (a & 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 0; if (a & 10) { return 3; } return 0; }";
assert 0 "int main() { int a = 0; if (a & 0) { return 3; } return 0; }";
assert 0 "int main() { int a = 9; if (a & 0) { return 3; } return 0; }";

echo -e "\nOK"