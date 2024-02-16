#include "types.h"
#include "defs.h"
#include "defines.h"

__attribute__((aligned(16))) char stack0[4096];
int main(void);
void timervec();

uint64 tmscratch[32];

void start() {
        // set prev to supervisor
        asm volatile("csrc mstatus, %0" ::"r"(3 << 11));
        asm volatile("csrs mstatus, %0" ::"r"(1 << 11));

        // supervisor entry
        asm volatile("csrw mepc, %0" ::"r"(main));

        // turn off paging
        asm volatile("csrw satp, %0" : : "r"(0));

        // delegate software interrupts to S-mode
        asm volatile("csrw mideleg, %0" : : "r"(0xff));  // MSIE, SSIE
        // enable software interrupts in S-mode
        asm volatile("csrw sie, %0" ::"r"(1 << 1));  // sie.SSIE

        // configure physical memory protection to give supervisor access to all memory
        asm volatile("csrw pmpaddr0, %0" ::"r"(0x3fffffffffffffULL));
        asm volatile("csrw pmpcfg0, %0" ::"r"(0xf));

        // timer interrupt
        *(uint64 *)CLINT_MTIMECMP = *(uint64 *)CLINT_MTIME + INTERVAL;
        asm volatile("csrs mstatus, %0" ::"r"(1 << 3));  // mstatus.MIE
        asm volatile("csrs mie, %0" ::"r"(1 << 7));      // mie.MTIE
        asm volatile("csrw mtvec, %0" ::"r"(timervec));

        // set up scratch area for M-mode trap handling
        uint64 *mscratch = &tmscratch[0];
        asm volatile("csrw mscratch, %0" ::"r"((uint64)mscratch));

        // switch to supervisor
        asm volatile("mret");
}