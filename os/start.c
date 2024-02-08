#include "defs.h"

__attribute__((aligned(512))) char stack0[4096];
int main(void);
void timervec();

unsigned long tscratch[32];

// core local interruptor (CLINT), which contains the timer
#define CLINT 0x2000000L
#define CLINT_MTIMECMP (CLINT + 0x4000)
#define CLINT_MTIME (CLINT + 0xBFF8)  // cycles since boot

void start() {
        // set prev to supervisor
        asm volatile("csrc mstatus, %0" ::"r"(0b11 << 11));
        asm volatile("csrs mstatus, %0" ::"r"(0b01 << 11));

        // supervisor entry
        asm volatile("csrw mepc, %0" ::"r"(main));

        // configure physical memory protection to give supervisor access to all memory
        asm volatile("csrw pmpaddr0, %0" ::"r"(0x3fffffffffffffULL));
        asm volatile("csrw pmpcfg0, %0" ::"r"(0xf));

        // timer interrupt (for now, without any side effects and S-mode involvement)
        int interval = 10000000;  // cycles; about one second in qemu.
        *(unsigned long *)CLINT_MTIMECMP = *(unsigned long *)CLINT_MTIME + interval;
        asm volatile("csrs mstatus, %0" ::"r"(0b1 << 3));  // mstatus.MIE
        asm volatile("csrs mie, %0" ::"r"(0b1 << 7));      // mie.MTIE
        asm volatile("csrw mtvec, %0" ::"r"(timervec));

        unsigned long *scratch = &tscratch[0];
        asm volatile("csrw mscratch, %0" ::"r"((unsigned long)scratch));

        // switch to supervisor
        asm volatile("mret");
}

void timertrap() {
        print("timer interval\n");
        int interval = 10000000;
        *(unsigned long *)CLINT_MTIMECMP = *(unsigned long *)CLINT_MTIME + interval;
}