#include "defs.h"

void kernelvec();

void kerneltrap() {
        print("kerneltrap\n");
        // acknowledge software interrupt
        asm volatile("csrc sip, %0" ::"r"(2));
}

void trapinit() {
        // install kernel trap vec
        asm volatile("csrw stvec, %0" : : "r"(kernelvec));
}

void intr_on() {
        // enable S-mode interrupts
        asm volatile("csrs sstatus, %0" : : "r"(1 << 1));  // SIE
}

void timertrap() {
        print("timer interval\n");
        *(unsigned long *)CLINT_MTIMECMP += INTERVAL;  // update mtimecmp
        asm volatile("csrs sip, %0" ::"r"(1 << 1));    // raise S-mode software interrupt
}