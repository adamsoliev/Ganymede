#!/bin/bash

assert() {
    expected="$(( ($1 % 256 + 256) % 256 ))" # in C, main's return value range (0 - 255)
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

assert 56 "int main() { int a = 23; if (a + a + 10) { return a + a + 10; } return 0; }";
assert 36 "int main() { int a = 23; if (a + a - 10) { return a + a - 10; } return 0; }";
# assert 10 "int main() { int a = 3; if (40 - a * 10) { return a + a - 10; } return 0; }";

assert 33 "int main() { int a = 23; if (a + 10) { return a + 10; } return 0; }";
assert 13 "int main() { int a = 23; if (a - 10) { return a - 10; } return 0; }";
assert 130 "int main() { int a = 13; if (a * 10) { return a * 10; } return 0; }";
assert 13 "int main() { int a = 130; if (a / 10) { return a / 10; } return 0; }";

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

assert 3 "int main() { int a = 9; if (a ^ 10) { return 3; } return 0; }";
assert 3 "int main() { int a = 0; if (a ^ 10) { return 3; } return 0; }";
assert 3 "int main() { int a = 9; if (a ^ 0) { return 3; } return 0; }";
assert 0 "int main() { int a = 0; if (a ^ 0) { return 3; } return 0; }";
assert 29 "int main() { int a = 23; if (a ^ 10) { return a ^ 10; } return 0; }";
assert 10 "int main() { int a = 0; if (a ^ 10) { return a ^ 10; } return 0; }";

assert 3 "int main() { int a = 23; if (a << 4) { return 3; } return 0; }";
assert 0 "int main() { int a = 0; if (a << 4) { return 3; } return 0; }";
assert 368 "int main() { int a = 23; if (a << 4) { return a << 4; } return 0; }";
assert 384 "int main() { int a = 24; if (a << 4) { return a << 4; } return 0; }";
assert 48 "int main() { int a = 3; if (a << 4) { return a << 4; } return 0; }";
assert 0 "int main() { int a = 0; if (a << 4) { return a << 4; } return 0; }";

assert 3 "int main() { int a = 23; if (a >> 4) { return 3; } return 0; }";
assert 0 "int main() { int a = 23; if (a >> 6) { return 3; } return 0; }";
assert 0 "int main() { int a = 0; if (a  >> 4) { return 3; } return 0; }";
assert 1 "int main() { int a = 23; if (a >> 4) { return 23 >> 4; } return 0; }";
assert 0 "int main() { int a = 0; if (a  >> 4) { return a >> 4; } return 0; }";

echo -e "\nOK"
