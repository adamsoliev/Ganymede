#include "defs.h"

__attribute__((aligned(16))) char stack0[4096];
int main(void);
void timervec();

unsigned long tmscratch[32];
unsigned long tsscratch[32];

void start() {
        // set prev to supervisor
        asm volatile("csrc mstatus, %0" ::"r"(3 << 11));
        asm volatile("csrs mstatus, %0" ::"r"(1 << 11));

        // supervisor entry
        asm volatile("csrw mepc, %0" ::"r"(main));

        // configure physical memory protection to give supervisor access to all memory
        asm volatile("csrw pmpaddr0, %0" ::"r"(0x3fffffffffffffULL));
        asm volatile("csrw pmpcfg0, %0" ::"r"(0xf));

        // delegate software interrupts to S-mode
        asm volatile("csrw mideleg, %0" : : "r"(0xff));  // MSIE, SSIE
        // enable software interrupts in S-mode
        asm volatile("csrw sie, %0" ::"r"(1 << 1));  // sie.SSIE

        // timer interrupt
        *(unsigned long *)CLINT_MTIMECMP = *(unsigned long *)CLINT_MTIME + INTERVAL;
        asm volatile("csrs mstatus, %0" ::"r"(1 << 3));  // mstatus.MIE
        asm volatile("csrs mie, %0" ::"r"(1 << 7));      // mie.MTIE
        asm volatile("csrw mtvec, %0" ::"r"(timervec));

        // set up scratch area for M-mode trap handling
        unsigned long *mscratch = &tmscratch[0];
        asm volatile("csrw mscratch, %0" ::"r"((unsigned long)mscratch));

        // set up scratch area for S-mode trap handling
        unsigned long *sscratch = &tsscratch[0];
        asm volatile("csrw sscratch, %0" ::"r"((unsigned long)sscratch));

        // switch to supervisor
        asm volatile("mret");
}