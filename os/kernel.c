
// data structures
enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };

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

struct proc {
        enum procstate state;
        int pid;
        unsigned long kstack;
        unsigned long sz;
        struct context context;
        char name[16];
};

// os
#define NPROC 4

// risc-v
#define PGSIZE 4096  // bytes per page
#define MAXVA (1L << (9 + 9 + 9 + 12 - 1))

// memlayout
#define TRAMPOLINE (MAXVA - PGSIZE)
#define KSTACK(p) (TRAMPOLINE - ((p) + 1) * 2 * PGSIZE)

// since this OS runs on a single CPU, we just have global vars for
struct proc *cur_proc;       // currently running process
struct context cur_context;  // swtch() here to enter scheduler
struct proc proct[NPROC];    // process table

__attribute__((aligned(16))) char stack0[4096];

unsigned long timer_scratch[5];

void kernelvec();

void w_stvec(unsigned long x);
unsigned long r_sip();
void w_sip(unsigned long x);
unsigned long r_sstatus();
void w_sstatus(unsigned long x);
void procinit(void);
void swtch(struct context *, struct context *);

void scheduler(void);

// _entry jumps here
int main() {
        procinit();                         // init process table
        w_stvec((unsigned long)kernelvec);  // install kernel trap vector

        scheduler();
        return 0;
}

void scheduler(void) {
        struct proc *p;
        cur_proc = 0;

        for (;;) {
                // enable device interrupt
                w_sstatus(r_sstatus() | (1 << 1));

                for (p = proct; p < &proct[NPROC]; p++) {
                        if (p->state == RUNNABLE) {
                                p->state = RUNNING;
                                cur_proc = p;
                                swtch(&cur_context, &p->context);
                                cur_proc = 0;
                        }
                }
        }
}

//////////////
// PROCESS
//////////////
void procinit(void) {
        struct proc *p;
        for (p = proct; p < &proct[NPROC]; p++) {
                p->state = UNUSED;
                p->kstack = KSTACK((int)(p - proct));
        }
}

//////////////
// TRAP
//////////////
void kerneltrap() {
        // acknowledge software interrupt by clearing sip.SSIP
        w_sip(r_sip() & (~2));
}

// RISC-V
void w_stvec(unsigned long x) { asm volatile("csrw stvec, %0" : : "r"(x)); }
unsigned long r_sip() {
        unsigned long x;
        asm volatile("csrr %0, sip" : "=r"(x));
        return x;
}
void w_sip(unsigned long x) { asm volatile("csrw sip, %0" : : "r"(x)); }
unsigned long r_sstatus() {
        unsigned long x;
        asm volatile("csrr %0, sstatus" : "=r"(x));
        return x;
}
void w_sstatus(unsigned long x) { asm volatile("csrw sstatus, %0" : : "r"(x)); }

/*
userinit
        allocproc
                find unused proc

                mark it as used
                allocate trapframe
                create user page table with trampoline/trapframe
                set up new context
                        p->context.ra = forkret
                        p->context.sp = p->kstack + PGSIZE

        allocate a page for initcode's instrs and data and copy them into it

        prepare for the first 'return' from kernel to user by setting up trapframe
                p->trapframe->epc = 0           // 
                p->trapframe->sp = PGSIZE       // 
        
        p->name = "initcode"
        p->state = RUNNABLE
*/