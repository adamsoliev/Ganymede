#include "defs.h"

void main(void) {
        uartinit();  // uart
        kinit();     // kernel physical memory allocator
        kvminit();   // kernel virtual memory
        procinit();  // process table

        trapinit();  // kernel trap vector

        allocproc(1);
        allocproc(2);

        int num = 92873;
        printf("num: %d\r\n", num);
        printf("pointer: %p\r\n", &num);
        char greet[] = "hello\n";
        char greet1[] = "world\n";
        printf("string: %s", greet);
        printf("string1: %s", greet1);

        scheduler();
}