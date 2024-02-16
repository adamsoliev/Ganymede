#include "types.h"
#include "defs.h"

struct proc proc[NPROC];

void procinit(void) {
        for (struct proc *p = proc; p < &proc[NPROC]; p++) {
                p->state = UNUSED;
                uint64 pa = (uint64)kalloc();
                // grows down | messes up someone's kernel stack if overflows
                p->kstack = pa + PGSIZE;
        }
}

// od -t xC ../user/proc
unsigned char process1[] = {
        0x13, 0x01, 0x01, 0xfe, 0x23, 0x3c, 0x11, 0x00, 0x23, 0x38, 0x81, 0x00, 0x13,
        0x04, 0x01, 0x02, 0x13, 0x07, 0x00, 0x00, 0xb7, 0x47, 0x0f, 0x00, 0x93, 0x87,
        0xf7, 0x23, 0x37, 0x03, 0x00, 0x00, 0x93, 0x02, 0x03, 0x04, 0x6f, 0x00, 0x80,
        0x00, 0x1b, 0x07, 0x17, 0x00, 0xe3, 0xde, 0xe7, 0xfe, 0x33, 0x85, 0x02, 0x00,
        0x93, 0x08, 0x50, 0x00, 0x73, 0x00, 0x00, 0x00, 0x6f, 0xf0, 0x5f, 0xfd, 0x50,
        0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x20, 0x31, 0x21, 0x0a, 0x00,
};

unsigned char process2[] = {
        0x13, 0x01, 0x01, 0xfe, 0x23, 0x3c, 0x11, 0x00, 0x23, 0x38, 0x81, 0x00, 0x13,
        0x04, 0x01, 0x02, 0x13, 0x07, 0x00, 0x00, 0xb7, 0x47, 0x0f, 0x00, 0x93, 0x87,
        0xf7, 0x23, 0x37, 0x03, 0x00, 0x00, 0x93, 0x02, 0x03, 0x04, 0x6f, 0x00, 0x80,
        0x00, 0x1b, 0x07, 0x17, 0x00, 0xe3, 0xde, 0xe7, 0xfe, 0x33, 0x85, 0x02, 0x00,
        0x93, 0x08, 0x50, 0x00, 0x73, 0x00, 0x00, 0x00, 0x6f, 0xf0, 0x5f, 0xfd, 0x50,
        0x72, 0x6f, 0x63, 0x65, 0x73, 0x73, 0x20, 0x32, 0x21, 0x0a, 0x00,
};

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

extern char trampoline[];

uint64 *proc_pagetable(struct proc *p) {
        uint64 *upt = kalloc();
        memset(upt, 0, PGSIZE);

        kvmmap(upt, TRAMPOLINE, (uint64)trampoline, PGSIZE, PTE_R | PTE_X);
        kvmmap(upt, TRAPFRAME, (uint64)p->trapframe, PGSIZE, PTE_R | PTE_W);
        return upt;
}

void allocproc(int pid) {
        struct proc *p;
        for (p = proc; p < &proc[NPROC]; p++) {
                if (p->state == UNUSED) goto found;
        }
        return;

found:
        p->trapframe = kalloc();
        p->pagetable = proc_pagetable(p);

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