#include "defs.h"

__attribute__((aligned(128))) char stack0[8192];
int main(void);
void timervec();

// core local interruptor (CLINT), which contains the timer.
#define CLINT 0x2000000L
#define CLINT_MTIMECMP (CLINT + 0x4000)
#define CLINT_MTIME (CLINT + 0xBFF8)  // cycles since boot.

void start() {
        // set prev to supervisor
        asm volatile("csrc mstatus, %0" ::"r"(0b11 << 11));
        asm volatile("csrs mstatus, %0" ::"r"(0b01 << 11));

        // supervisor entry
        asm volatile("csrw mepc, %0" ::"r"(main));

        // configure physical memory protection to give supervisor access to all memory
        asm volatile("csrw pmpaddr0, %0" ::"r"(0x3fffffffffffffULL));
        asm volatile("csrw pmpcfg0, %0" ::"r"(0xf));

        /* PLAN (timer interrupt) | WITHOUT SOFTWARE INVOLVEMENT
                -- in start
                set mtimecmp
                set mstatus.MIE to enable M interrupts in general
                set mie.MTIE to enable M timer interrupts
                register a handler by writing 'mtvec'

                -- in the handler while trapped with timer interrupt
                save all regs
                print timer interrupt
                reset mtimecmp
                restore all regs
                mret
        */

        int interval = 10000000;  // cycles; about one second in qemu.
        *(unsigned long*)CLINT_MTIMECMP = *(unsigned long*)CLINT_MTIME + interval;
        asm volatile("csrs mstatus, %0" ::"r"(0b1 << 3));  // mstatus.MIE
        asm volatile("csrs mie, %0" ::"r"(0b1 << 7));      // mie.MTIE
        asm volatile("csrw mtvec, %0" ::"r"(timervec));

        // switch to supervisor
        asm volatile("mret");
}

void timertrap() {
        uartputc('t');
        uartputc('i');
        uartputc('m');
        uartputc('e');
        uartputc('r');
        uartputc('\n');
        int interval = 10000000;  // cycles; about one second in qemu.
        *(unsigned long*)CLINT_MTIMECMP = *(unsigned long*)CLINT_MTIME + interval;
}

int main(void) {
        uartinit();

        while (1) {
                for (int i = 0; i < 1000000; i++)
                    ;
                uartputc('h');
                uartputc('e');
                uartputc('l');
                uartputc('l');
                uartputc('o');
                uartputc('\n');
        }

        return 0;
}
