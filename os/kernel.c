/*
- assumption here is that we are running in a system with single core
- paging is off, hence we are dealing with direct memory addresses of qemu
*/
#include "kernel.h"

// data structures
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

struct trapframe {
        /*   0 */ unsigned long kernel_satp;    // kernel page table
        /*   8 */ unsigned long kernel_sp;      // top of process's kernel stack
        /*  16 */ unsigned long kernel_trap;    // usertrap()
        /*  24 */ unsigned long epc;            // saved user program counter
        /*  32 */ unsigned long kernel_hartid;  // saved kernel tp
        /*  40 */ unsigned long ra;
        /*  48 */ unsigned long sp;
        /*  56 */ unsigned long gp;
        /*  64 */ unsigned long tp;
        /*  72 */ unsigned long t0;
        /*  80 */ unsigned long t1;
        /*  88 */ unsigned long t2;
        /*  96 */ unsigned long s0;
        /* 104 */ unsigned long s1;
        /* 112 */ unsigned long a0;
        /* 120 */ unsigned long a1;
        /* 128 */ unsigned long a2;
        /* 136 */ unsigned long a3;
        /* 144 */ unsigned long a4;
        /* 152 */ unsigned long a5;
        /* 160 */ unsigned long a6;
        /* 168 */ unsigned long a7;
        /* 176 */ unsigned long s2;
        /* 184 */ unsigned long s3;
        /* 192 */ unsigned long s4;
        /* 200 */ unsigned long s5;
        /* 208 */ unsigned long s6;
        /* 216 */ unsigned long s7;
        /* 224 */ unsigned long s8;
        /* 232 */ unsigned long s9;
        /* 240 */ unsigned long s10;
        /* 248 */ unsigned long s11;
        /* 256 */ unsigned long t3;
        /* 264 */ unsigned long t4;
        /* 272 */ unsigned long t5;
        /* 280 */ unsigned long t6;
};

enum procstate { UNALLOCATED, RUNNABLE, RUNNING };

struct proc {
        enum procstate state;
        int pid;

        unsigned long kstack;
        unsigned long sz;
        struct trapframe *trapframe;
        struct context context;
        char name[16];
};

// trap
void kernelvec(void);
void usertrap(void);
void usertrapret(void);

unsigned long timer_scratch[5];

extern char trampoline[], uservec[];
void userret(void);

// process
void procinit(void);
void sched(void);
void yield(void);
void userinit(void);

// riscv
void w_stvec(unsigned long x);
unsigned long r_sip();
void w_sip(unsigned long x);
unsigned long r_sstatus();
void w_sstatus(unsigned long x);
unsigned long r_sepc();
void w_sepc(unsigned long x);

// string
void *memmove(void *dst, const void *src, unsigned int n);
char *safestrcpy(char *s, const char *t, int n);

// os
int main();
void scheduler(void);
void swtch(struct context *, struct context *);

__attribute__((aligned(16))) char stack0[4096];

// process
struct proc proct[NPROC];
struct proc *cur_proc;
struct context cur_context;

///////////////////////////////////////////////////////////////////////////////
// _entry jumps here
int main() {
        procinit();                         // process table
        w_stvec((unsigned long)kernelvec);  // install kernel trap vector
        userinit();                         // first user process

        scheduler();
        return 0;
}

void scheduler(void) {
        struct proc *p;
        cur_proc = 0;
        for (;;) {
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

////////////////////////////////
// PROCESS
////////////////////////////////

unsigned char initcode[] = {0x13, 0x03, 0x70, 0x01, 0x6f, 0xf0, 0xdf, 0xff};

void userinit(void) {
        struct proc *p = &proct[0];

        p->pid = 1;
        p->trapframe = (struct trapframe *)TRAPFRAME;  // START + 3MB

        // set up new context to start executing at usertrapret,
        // which returns to user space.
        p->context.ra = (unsigned long)usertrapret;
        p->context.sp = p->kstack + PGSIZE;

        // copy initcode's instructions to mem starting at 0x80500000
        // pc has to be set to the above too
        char *STARTADDR = (void *)0x80500000;  // START + 5MB
        char *copy = STARTADDR;
        for (int i = 0; i < 8; i++) {
                *copy = initcode[i];
                copy++;
        }
        p->sz = PGSIZE;

        // prepare for the very first 'return' from kernel to user
        p->trapframe->epc = (unsigned long)STARTADDR;              // user pc
        p->trapframe->sp = (unsigned long)STARTADDR + PGSIZE * 2;  // user stack pointer

        safestrcpy(p->name, "initcode", sizeof(p->name));

        p->state = RUNNABLE;
}

void procinit(void) {
        struct proc *p;
        for (p = proct; p < &proct[NPROC]; p++) {
                p->state = UNALLOCATED;
                p->kstack = KSTACK((int)(p - proct));
        }
}

void yield(void) {
        struct proc *p = cur_proc;
        p->state = RUNNABLE;
        sched();
}

void sched(void) {
        struct proc *p = cur_proc;
        swtch(&p->context, &cur_context);
}

////////////////////////////////
// TRAP
////////////////////////////////

// handle an interrupt, exception, or system call from user space
void usertrap(void) {
        // send interrupts and exceptions to kerneltrap(),
        // since we're now in the kernel.
        w_stvec((unsigned long)kernelvec);

        struct proc *p = cur_proc;

        // save user program counter.
        p->trapframe->epc = r_sepc();

        // acknowledge the software interrupt by clearing
        // the SSIP bit in sip.
        w_sip(r_sip() & ~2);

        // give up the CPU since this is a timer interrupt
        yield();

        usertrapret();
}

// return to user space
void usertrapret(void) {
        struct proc *p = cur_proc;

        // we're about to switch the destination of traps from
        // kerneltrap() to usertrap(), so turn off interrupts until
        // we're back in user space, where usertrap() is correct.
        w_sstatus(r_sstatus() & ~(1 << 1));

        // send syscalls, interrupts, and exceptions to uservec in trampoline.S
        w_stvec((unsigned long)uservec);

        // set up trapframe values that uservec will need when
        // the process next traps into the kernel.
        p->trapframe->kernel_sp = p->kstack + PGSIZE;  // process's kernel stack
        p->trapframe->kernel_trap = (unsigned long)usertrap;

        // set up the registers that trampoline.S's sret will use
        // to get to user space.

        // set S Previous Privilege mode to User.
        unsigned long x = r_sstatus();
        x &= ~(1 << 8);  // clear SPP to 0 for user mode
        x |= (1 << 5);   // enable interrupts in user mode
        w_sstatus(x);

        // set S Exception Program Counter to the saved user pc.
        w_sepc(p->trapframe->epc);

        // jump to userret in trampoline.S at the top of memory, which
        // restores user registers, and switches to user mode with sret.
        userret();
}

void kerneltrap() {
        // acknowledge software interrupt by clearing sip.SSIP
        w_sip(r_sip() & (~2));
}

////////////////////////////////
// RISC-V
////////////////////////////////
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
unsigned long r_sepc() {
        unsigned long x;
        asm volatile("csrr %0, sepc" : "=r"(x));
        return x;
}
void w_sepc(unsigned long x) { asm volatile("csrw sepc, %0" : : "r"(x)); }

////////////////////////////////
// STRING
////////////////////////////////
// Like strncpy but guaranteed to NUL-terminate.
char *safestrcpy(char *s, const char *t, int n) {
        char *os;

        os = s;
        if (n <= 0) return os;
        while (--n > 0 && (*s++ = *t++) != 0)
                ;
        *s = 0;
        return os;
}
