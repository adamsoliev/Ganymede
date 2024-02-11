#include "defs.h"

int main(void) {
        uartinit();  // uart
        kinit();     // kernel physical allocator
        kvminit();   // kernel virtual memory
        procinit();  // process table

        trapinit();  // install kernel vector trap

        while (1) {
                intr_on();

                for (int i = 0; i < 100000000; i++)
                        ;
                print("HELLO WORLD!\n");
        }

        return 0;
}