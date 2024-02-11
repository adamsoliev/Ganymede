#include "defs.h"

void kernelvec();

void kerneltrap() {
        print("kerneltrap\n");
        asm volatile("csrc sip, %0" ::"r"(2));
}

void trapinit() { asm volatile("csrw stvec, %0" : : "r"(kernelvec)); }

void intr_on() {
        // enable S-mode interrupts
        asm volatile("csrw sstatus, %0" : : "r"(1 << 1));  // SIE
}

void timertrap() {
        print("timer interval\n");
        *(unsigned long *)CLINT_MTIMECMP += INTERVAL;
        asm volatile("csrw sip, %0" ::"r"(2));
}