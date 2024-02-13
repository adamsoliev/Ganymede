#include "defs.h"

#define NPROC 2

// Saved registers for kernel context switches.
struct context {
        unsigned long ra;
        unsigned long sp;

        // callee-saved
        unsigned long s0;
        unsigned long s1;
        unsigned long s2;
        unsigned long s3;
        unsigned long s4;
        unsigned long s5;
        unsigned long s6;
        unsigned long s7;
        unsigned long s8;
        unsigned long s9;
        unsigned long s10;
        unsigned long s11;
};

enum procstate { UNUSED, RUNNABLE, RUNNING };

// Per-process state
struct proc {
        enum procstate state;    // Process state
        int pid;                 // Process ID
        unsigned long kstack;    // Virtual address of kernel stack
        unsigned long sz;        // Size of process memory (bytes)
        struct context context;  // swtch() here to run process
        char name[16];           // Process name (debugging)
};

struct proc proc[NPROC];

void procinit(void) {
        for (struct proc* p = proc; p < &proc[NPROC]; p++) {
                p->state = UNUSED;
                unsigned long pa = (unsigned long)kalloc();
                // grows down
                p->kstack = pa + PGSIZE;
        }
}