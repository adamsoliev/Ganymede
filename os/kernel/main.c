#include "types.h"
#include "defs.h"

void main(void) {
        uartinit();  // uart
        kinit();     // kernel physical memory allocator
        kvminit();   // kernel virtual memory
        procinit();  // process table

        trapinit();  // kernel trap vector

        allocproc(1);
        allocproc(2);

        scheduler();
}