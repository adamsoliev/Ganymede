#include "defs.h"

#define NPROC 5

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

enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };

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
        for (struct proc *p = proc; p < &proc[NPROC]; p++) {
                p->state = UNUSED;
                p->kstack = KSTACK((int)(p - proc));
        }
}

void proc_mapstack(unsigned long *ptable) {
        for (struct proc *p = proc; p < &proc[NPROC]; p++) {
                char *pa = kalloc();
                if (pa == 0) panic("proc_mapstack: kalloc\n");
                unsigned long va = KSTACK((int)(p - proc));
                kvmmap(ptable, va, (unsigned long)pa, PGSIZE, PTE_R | PTE_W);
        }
}