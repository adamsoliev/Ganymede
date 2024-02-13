#include "defs.h"
#include "types.h"

struct proc proc[NPROC];

void procinit(void) {
        for (struct proc *p = proc; p < &proc[NPROC]; p++) {
                p->state = UNUSED;
                uint64 pa = (uint64)kalloc();
                // grows down | messes up someone's kernel stack if overflows
                p->kstack = pa + PGSIZE;
        }
}

void proc1() {
        while (1) {
                for (int i = 0; i < 100000000; i++)
                        ;
                print("PROC1\n");
        }
}

void proc2() {
        while (1) {
                for (int i = 0; i < 100000000; i++)
                        ;
                print("PROC2\n");
        }
}

void allocproc(int pid) {
        struct proc *p;
        for (p = proc; p < &proc[NPROC]; p++) {
                if (p->state == UNUSED) goto found;
        }
        return;

found:
        p->state = RUNNABLE;
        memset(&p->context, 0, sizeof(p->context));
        if (pid == 1) {
                p->pid = 1;
                p->context.ra = (uint64)proc1;
        } else {
                p->pid = 2;
                p->context.ra = (uint64)proc2;
        }
        p->context.sp = p->kstack + PGSIZE;
}

struct context cur_context;
struct proc *cur_proc;

void scheduler(void) {
        while (1) {
                intr_on();
                for (struct proc *p = proc; p < &proc[NPROC]; p++) {
                        if (p->state == RUNNABLE) {
                                p->state = RUNNING;
                                cur_proc = p;
                                swtch(&cur_context, &p->context);
                                cur_proc = 0;
                        }
                }
        }
}

void yield(void) {
        cur_proc->state = RUNNABLE;
        swtch(&cur_proc->context, &cur_context);
}