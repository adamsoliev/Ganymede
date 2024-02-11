#include "defs.h"

// TODO: investigate why smaller values aren't working
__attribute__((aligned(16))) char stack0[4096];
int main(void);
void timervec();

unsigned long tmscratch[32];
unsigned long tsscratch[32];

// core local interruptor (CLINT), which contains the timer
#define CLINT 0x2000000L
#define CLINT_MTIMECMP (CLINT + 0x4000)
#define CLINT_MTIME (CLINT + 0xBFF8)  // cycles since boot

#define INTERVAL 10000000

void start() {
        // set prev to supervisor
        asm volatile("csrc mstatus, %0" ::"r"(0b11 << 11));
        asm volatile("csrs mstatus, %0" ::"r"(0b01 << 11));

        // supervisor entry
        asm volatile("csrw mepc, %0" ::"r"(main));

        // configure physical memory protection to give supervisor access to all memory
        asm volatile("csrw pmpaddr0, %0" ::"r"(0x3fffffffffffffULL));
        asm volatile("csrw pmpcfg0, %0" ::"r"(0xf));

        // delegate software interrupts to S-mode
        asm volatile("csrw mideleg, %0" : : "r"(0xff));  // MTIE, STIE, MSIE, SSIE
        asm volatile("csrw medeleg, %0" : : "r"(0xff));  // MTIE, STIE, MSIE, SSIE
        // enable software interrupts in S-mode
        asm volatile("csrw sie, %0" ::"r"(1 << 1));  // sie.STIE sie.SSIE

        // timer interrupt (for now, without any side effects and S-mode involvement)
        *(unsigned long *)CLINT_MTIMECMP = *(unsigned long *)CLINT_MTIME + INTERVAL;
        asm volatile("csrs mstatus, %0" ::"r"(0b1 << 3));  // mstatus.MIE
        asm volatile("csrs mie, %0" ::"r"(0b1 << 7));      // mie.MTIE
        asm volatile("csrw mtvec, %0" ::"r"(timervec));

        unsigned long *mscratch = &tmscratch[0];
        asm volatile("csrw mscratch, %0" ::"r"((unsigned long)mscratch));

        unsigned long *sscratch = &tsscratch[0];
        asm volatile("csrw sscratch, %0" ::"r"((unsigned long)sscratch));

        // switch to supervisor
        asm volatile("mret");
}

void timertrap() {
        print("timer interval\n");
        *(unsigned long *)CLINT_MTIMECMP += INTERVAL;
        asm volatile("csrw sip, %0" ::"r"(2));
}