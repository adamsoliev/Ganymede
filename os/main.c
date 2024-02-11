#include "defs.h"

int main(void) {
        uartinit();  // uart
        kinit();     // kernel physical memory allocator
        kvminit();   // kernel virtual memory
        procinit();  // process table

        trapinit();  // kernel trap vector 

        while (1) {
                intr_on();

                for (int i = 0; i < 100000000; i++)
                        ;
                print("HELLO WORLD!\n");
        }

        return 0;
}