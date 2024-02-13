#include "defs.h"

int main(void) {
        uartinit();  // uart
        kinit();     // kernel physical memory allocator
        kvminit();   // kernel virtual memory
        procinit();  // process table

        trapinit();  // kernel trap vector

        allocproc(1);
        allocproc(2);

        while (1) {
                intr_on();

                for (int i = 0; i < 100000000; i++)
                        ;
                print("HELLO WORLD!\n");
        }

        return 0;
}