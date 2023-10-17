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

assert 17 "int sum(int ab, int ba, int ca) { return ab + ba + ca; } int main() { int a = sum(4, 9, 4); return a; }";
assert 13 "int sum(int ab, int ba) { return ab + ba; } int main() { int a = sum(4, 9); return a; }";
assert 12 "int func(int ab) { return ab * 3; } int main() { int a = func(4); return a; }";

assert 23 "int main() { int a = 23; if (a > 22) goto Lll; Lll: return a; return 0; }"
assert 23 "int main() { int a = 23; goto Lll; Lll: return a; return 0; }"

assert 90 "int main() { int a = 1; switch (a) { case 1: { a = a * 90; break;} case 2: { a = a * 7; break;} default: { a = a * 2;}} return a; }";
assert 90 "int main() { int a = 1; switch (a) { case 1: { a = a * 90; break;} case 2: { a = a * 7; break;} default: a = a * 2;} return a; }";
assert 70 "int main() { int a = 1; switch (a) { case 1: { a = a * 10;} case 2: { a = a * 7; break;} default: a = a * 2;} return a; }";
assert 140 "int main() { int a = 1; switch (a) { case 1: { a = a * 10;} case 2: { a = a * 7;} default: a = a * 2;} return a; }";
assert 10 "int main() { int a = 1; switch (a) { case 1: { a = a * 10;} } return a; }";
assert 2 "int main() { int a = 1; switch (a) { default: a = a * 2;} return a; }";
assert 1 "int main() { int a = 1; switch (a) {} return a; }";
assert 90 "int main() { int a = 1; switch (a) { case 1: a = a * 90; break; case 2: a = a * 7; break; default: a = a * 2;} return a; }";
assert 40 "int main() { int a = 2; switch (a) { case 1: a = a * 90; break; case 2: a = a * 20; break; default: a = a * 2;} return a; }";
assert 6 "int main() { int a = 3; switch (a) { case 1: a = a * 90; break; case 2: a = a * 7; break; default: a = a * 2;} return a; }";
assert 10 "int main() { int a = 1; do a++; while (a < 10); return a; }";
assert 16 "int main() { int a = 1; do { a = a * 2; } while (a < 10); return a; }";
assert 11 "int main() { int a = 1; int i = 0; for (; i < 10; i++) { a++; } return a; }";
assert 10 "int main() { int a = 1; int i = 0; for (; i < 10; i++) { if (i == 3) continue; a++; } return a; }";
assert 1 "int main() { int a = 1; int i = 0; for (; i < 10; i++) ; return a; }";
assert 0 "int main() { int a = 23; if (a) ; else return 0; }";
assert 11 "int main() { int a = 1; for (int i = 0; i < 10; i++) { a++; } return a; }";
assert 11 "int main() { int a = 1; int i = 3; for (i = 0; i < 10; i++) { a++; } return a; }";

assert 23 "int main() { int a = 0; while (a <= 20) { if (a % 2 == 0) { a = a + 3; } else { a = a + 10; } } return a; }";
assert 40 "int main() { int a = 0; while (a <= 20) { int b = a * 2 + 1; a = a + b; } return a; }";
assert 10 "int main() { int a = 0; while (a < 10) { a = a + 2; } return a; }";
assert 20 "int main() { int a = 0; while (a < 10) { a = a + 2; if (a == 2) continue; a = a * 2; } return a; }";
assert 12 "int main() { int a = 0; while (a < 10) { a = a + 2; a = a * 2; } return a; }";

assert 23 "int main() { int a = 23; if (a) return 23; else return 0; }";
assert 23 "int main() { int a = 24; if (a > 23) return 23; else if (a == 23) return 10; else return 0; }";
assert 10 "int main() { int a = 23; if (a > 23) return 23; else if (a == 23) return 10; else return 0; }";
assert 0 "int main() { int a = 22; if (a > 23) return 23; else if (a == 23) return 10; else return 0; }";

assert 23 "int main() { int a; a = 23; return a; }";
assert 46 "int main() { int a; int b; a = 23; b = a * 2; return b; }";
assert 24 "int main() { int a = 23; ++a; return a; }";
assert 46 "int main() { int a = 23; int b = a * 2; return b; }";
assert 36 "int main() { int a = 23; int b = a * 2 + 10 * 2 - 30; return b; }";
assert 6 "int main() { int a = 23; int b = a * 2 + 10 * 2 - 30; int c = b - 30; return c; }";
assert 22 "int main() { int a = 23; --a; return a; }";
assert 7 "int main() { int a = 23; -a; return a + 30; }";
assert 23 "int main() { int a = 23; +a; return a; }";
assert 24 "int main() { int a = 23; a++; return a; }";
assert 23 "int main() { int a = 23; return a++; }";
assert 23 "int main() { int a = 23; return a--; }";

assert 23 "int main() { int a = 23; if (a) return 23; return 0; }";

assert 23 "int main() { int a = 23; if (a) { return 23; }  return 0; }";
assert 0 "int main() { int a = 23; if (!a) { return 23; }  return 0; }";
assert 26 "int main() { int a = 23; a = a - 50; return ~a; }";

assert 100 "int main() { int a = 23; a = a > 10 ? 100 : 1; return a; }";
assert 1 "int main() { int a = 23; a = a < 10 ? 100 : 1; return a; }";
assert 9 "int main() { int a = 23; a = a == 10 ? 100 : 1; return a * 9; }";

assert 26 "int main() { int a = 23; a = a + 3; return a; }";
assert 55 "int main() { int a = 23; int b = 32; a = a + b; return a; }";
assert 12 "int main() { int a = 3; int b = 2; int c = 5; a = a + b; b = b + c;  return a + b; }";
assert 55 "int main() { int a = 23; int b = 32; if (a + b) { b = a + b; } return b; }";

assert 56 "int main() { int a = 23; if (a + a + 10) { return a + a + 10; } return 0; }";
assert 36 "int main() { int a = 23; if (a + a - 10) { return a + a - 10; } return 0; }";
assert 10 "int main() { int a = 3; if (40 - a * 10) { return 40 - a * 10; } return 0; }";
assert 70 "int main() { int a = 3; if (40 + a * 10) { return 40 + a * 10; } return 0; }";
assert 29 "int main() { int a = 3; if (4 + a * 10 - 5) { return 4 + a * 10 - 5; } return 0; }";
assert 32 "int main() { int a = 3; if (4 + a * 10 - 6 / 3) { return 4 + a * 10 - 6 / 3; } return 0; }";

assert 33 "int main() { int a = 23; if (a + 10) { return a + 10; } return 0; }";
assert 13 "int main() { int a = 23; if (a - 10) { return a - 10; } return 0; }";
assert 130 "int main() { int a = 13; if (a * 10) { return a * 10; } return 0; }";
assert 13 "int main() { int a = 130; if (a / 10) { return a / 10; } return 0; }";
assert 7 "int main() { int a = 137; if (a % 10) { return a % 10; } return 0; }";

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
assert 7 "int main() { int a = 9; if (5 & 3 | 7) { return 5 & 3 | 7; } return 0; }";

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
assert 40 "int main() { int a = 5; if (a << 3 & 60) { return a << 3 & 60; } return 0; }";

assert 3 "int main() { int a = 23; if (a >> 4) { return 3; } return 0; }";
assert 0 "int main() { int a = 23; if (a >> 6) { return 3; } return 0; }";
assert 0 "int main() { int a = 0; if (a  >> 4) { return 3; } return 0; }";
assert 1 "int main() { int a = 23; if (a >> 4) { return 23 >> 4; } return 0; }";
assert 0 "int main() { int a = 0; if (a  >> 4) { return a >> 4; } return 0; }";

echo -e "\nOK"
