#include "defs.h"

__attribute__ ((aligned (128))) char stack0[4096];
int main(void);

void start() {
        // set prev to supervisor
        unsigned long mstatus;
        asm volatile("csrr %0, mstatus" : "=r"(mstatus));
        asm volatile("csrw mstatus, %0" ::"r"((mstatus & ~(3L << 11)) | (1L << 11)));

        // supervisor entry
        asm volatile("csrw mepc, %0" ::"r"(main));

        // configure physical memory protection to give supervisor access to all memory
        asm volatile("csrw pmpaddr0, %0" ::"r"(0x3fffffffffffffULL));
        asm volatile("csrw pmpcfg0, %0" ::"r"(0xf));

        // switch to supervisor
        asm volatile("mret");
}

int main(void) {
        uartinit();

        uartputc('h');
        uartputc('e');
        uartputc('l');
        uartputc('l');
        uartputc('o');
        uartputc('\n');

        return 0;
}
