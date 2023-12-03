#include <stddef.h>
#include <stdint.h>

typedef unsigned long uint64;

#define MSTATUS_MPP_MASK (3L << 11)  // previous mode.
#define MSTATUS_MPP_M (3L << 11)
#define MSTATUS_MPP_S (1L << 11)
#define MSTATUS_MPP_U (0L << 11)
#define MSTATUS_MIE (1L << 3)  // machine-mode interrupt enable.

// entry.S needs one stack
__attribute__((aligned(16))) char stack0[4096];

void main();

unsigned char *uart = (unsigned char *)0x10000000;
void putchar(char c) {
        *uart = c;
        return;
}

void print(const char *str) {
        while (*str != '\0') {
                putchar(*str);
                str++;
        }
        return;
}

void main(void) { print("Hello world!\n"); }