#include "defs.h"

void main();
void timerinit();
void timervec();

// only > 128 works, not sure why
__attribute__ ((aligned (128))) char stack0[4096];
unsigned long timer_scratch[5];

void start() {
    // set prev to supervisor
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    asm volatile("csrw mstatus, %0" :: "r"((mstatus & ~(3L << 11)) | (1L << 11)));

    // supervisor entry
    asm volatile("csrw mepc, %0" ::"r"(main));

    // turn off paging
    asm volatile("csrw satp, %0" ::"r"(0));

    // delegate all S/U exceptions, interrupts to supervisor
    asm volatile("csrw medeleg, %0" ::"r"(0xffff));
    asm volatile("csrw mideleg, %0" ::"r"(0xffff));
    unsigned long sie;
    asm volatile("csrr %0, sie" : "=r"(sie));
    asm volatile("csrw sie, %0" :: "r"((sie | (1L << 9) | (1L << 5) | (1L << 1))));

    // configure physical memory protection to give supervisor access to all memory
    asm volatile("csrw pmpaddr0, %0" :: "r"(0x3fffffffffffffULL));
    asm volatile("csrw pmpcfg0, %0" :: "r"(0xf));

    timerinit();

    // switch to supervisor
    asm volatile("mret");
}

void timerinit() {
    // configure timer interrupt
    int interval = 1000000;
    *(unsigned long*)CLINT_MTIMECMP = *(unsigned long*)CLINT_MTIME + interval;

    unsigned long *scratch = &timer_scratch[0];
    scratch[3] = CLINT_MTIMECMP;
    scratch[4] = interval;
    asm volatile("csrw mscratch, %0" :: "r"((unsigned long)scratch));

    // M trap handler
    asm volatile("csrw mtvec, %0" :: "r"(timervec));
    // enable M interrupts
    unsigned long mstatus;
    asm volatile("csrr %0, mstatus" : "=r"(mstatus));
    asm volatile("csrw mstatus, %0" :: "r"((mstatus | (1L << 3)))); // mie
    // enable M timer interrupts
    unsigned long mie;
    asm volatile("csrr %0, mie" : "=r"(mie));
    asm volatile("csrw mie, %0" :: "r"((mie | (1L << 7)))); // mtie
}