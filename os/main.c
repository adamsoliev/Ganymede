#include "defs.h"

__attribute__((aligned(128))) char stack0[4096];
int main(void);

void start() {
        // set prev to supervisor
        asm volatile("csrc mstatus, %0" ::"r"(0b11 << 11));
        asm volatile("csrs mstatus, %0" ::"r"(0b01 << 11));

        // supervisor entry
        asm volatile("csrw mepc, %0" ::"r"(main));

        // configure physical memory protection to give supervisor access to all memory
        asm volatile("csrw pmpaddr0, %0" ::"r"(0x3fffffffffffffULL));
        asm volatile("csrw pmpcfg0, %0" ::"r"(0xf));

        // timer interrupt
        // asm volatile("csrs mstatus, %0" :: "r" (0b10)); // SIE
        // asm volatile("csrs mie, %0" :: "r" (0b101 << 5)); // MTIE & STIE
        // asm volatile("csrs mideleg, %0" :: "r" (0b1 << 5)); // STIE

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
