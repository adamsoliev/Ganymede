#include "defs.h"

void main();
void timervec();

// core local interruptor (CLINT), which contains the timer.
#define CLINT 0x2000000L
#define CLINT_MTIMECMP (CLINT + 0x4000)
#define CLINT_MTIME (CLINT + 0xBFF8) 
#define INTERVAL 1000000

// only > 128 works, not sure why
__attribute__ ((aligned (128))) char stack0[4096];

void start() {
    // set prev to supervisor
    int mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    asm volatile("csrw mstatus, %0" :: "r"((mstatus & ~(3L << 11)) | (1L << 11)));

    // supervisor entry
    asm volatile("csrw mepc, %0" ::"r"(main));

    // turn off paging
    asm volatile("csrw satp, %0" ::"r"(0));

    // delegate all S/U exceptions, interrupts to supervisor
    asm volatile("csrw medeleg, %0" ::"r"(0xffff));
    asm volatile("csrw mideleg, %0" ::"r"(0xffff));
    int sie;
    asm volatile("csrr %0, sie" : "=r"(sie));
    asm volatile("csrw sie, %0" :: "r"((sie | (1L << 9) | (1L << 5) | (1L << 1))));

    // configure physical memory protection to give supervisor access to all memory
    asm volatile("csrw pmpaddr0, %0" :: "r"(0x3fffffffffffffULL));
    asm volatile("csrw pmpcfg0, %0" :: "r"(0xf));

    // configure timer interrupt
    *(unsigned long*)CLINT_MTIMECMP = *(unsigned long*)CLINT_MTIME + INTERVAL;
    asm volatile("csrw mtvec, %0" :: "r"(timervec));
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    asm volatile("csrw mstatus, %0" :: "r"((mstatus & (1L << 3)))); // mie
    int mie;
    asm volatile("csrr %0, mie" : "=r"(mie));
    asm volatile("csrw mie, %0" :: "r"((mie & (1L << 7)))); // mtie

    // switch to supervisor
    asm volatile("mret");
}

void timervec() {
    printf("timer interrupt\n");
    *(unsigned long*)CLINT_MTIMECMP = *(unsigned long*)CLINT_MTIME + INTERVAL;
    asm volatile("csrw sip, %0" :: "r"(2)); // supervisor software interrupt
    asm volatile("mret");
}