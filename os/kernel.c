
// data structures
// per-process data for the trap handling code in trampoline.S.
// sits in a page by itself just under the trampoline page in the
// user page table. not specially mapped in the kernel page table.
// uservec in trampoline.S saves user registers in the trapframe,
// then initializes registers from the trapframe's
// kernel_sp, kernel_hartid, kernel_satp, and jumps to kernel_trap.
// usertrapret() and userret in trampoline.S set up
// the trapframe's kernel_*, restore user registers from the
// trapframe, switch to the user page table, and enter user space.
// the trapframe includes callee-saved user registers like s0-s11 because the
// return-to-user path via usertrapret() doesn't return through
// the entire kernel call stack.
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
        struct trapframe *trapframe;  // data page for trampoline.S
        struct context context;
        char name[16];
};

// os
#define NPROC 4

// risc-v
#define PGSIZE 4096  // bytes per page
#define MAXVA (1L << (9 + 9 + 9 + 12 - 1))

#define MEMSTART 0x80000000
#define MEMEND 0x88000000

// memlayout
#define TRAMPOLINE (MAXVA - PGSIZE)
#define KSTACK(p) (TRAMPOLINE - ((p) + 1) * 2 * PGSIZE)

// since this OS runs on a single CPU, we just have global vars for
struct proc *cur_proc;       // currently running process
struct context cur_context;  // swtch() here to enter scheduler
struct proc proct[NPROC];    // process table

__attribute__((aligned(16))) char stack0[4096];

unsigned long timer_scratch[5];
char end[];  // first address after kernel; defined in virl.ld

void kernelvec();
void uservec();
void userret();
void userinit(void);
void usertrapret(void);
void usertrap(void);

void w_stvec(unsigned long x);
unsigned long r_sip();
void w_sip(unsigned long x);
unsigned long r_sstatus();
void w_sstatus(unsigned long x);
void w_sepc(unsigned long x);
void procinit(void);
void swtch(struct context *, struct context *);

void scheduler(void);

// _entry jumps here
int main() {
        procinit();                         // init process table
        w_stvec((unsigned long)kernelvec);  // install kernel trap vector

        userinit();

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
// USER
//////////////
void userinit(void) {
        struct proc *p = &proct[0];
        p->pid = 1;
        p->state = USED;

        p->context.ra = (unsigned long)usertrapret;
        p->context.sp = p->kstack + PGSIZE;

        // TODO: copy instructions/data into this process's page
        p->sz = PGSIZE;

        // first 'return' from kernel to user
        // p->trapframe->epc = 0;
        // p->trapframe->sp = PGSIZE;

        // clang-format off
        p->name[0] = 'u'; p->name[1] = 's'; p->name[2] = 'e'; p->name[3] = 'r'; p->name[4] = '1'; p->name[15] = '\0';
        // clang-format on
        p->state = RUNNABLE;
}

void usertrapret(void) {
        struct proc *p = cur_proc;

        // we're about to switch the destination of traps from
        // kerneltrap() to usertrap(), so turn off interrupts until
        // we're back in user space, where usertrap() is correct.
        w_sstatus(r_sstatus() & ~(1 << 1));

        w_stvec((unsigned long)uservec);

        // set up trapframe values that uservec will need when
        // the process next traps into the kernel.
        // p->trapframe->kernel_satp = r_satp();          // kernel page table
        p->trapframe->kernel_sp = p->kstack + PGSIZE;  // process's kernel stack
        p->trapframe->kernel_trap = (unsigned long)usertrap;
        // p->trapframe->kernel_hartid = r_tp();  // hartid for cpuid()

        // set up the registers that trampoline.S's sret will use
        // to get to user space.

        // set S Previous Privilege mode to User.
        unsigned long x = r_sstatus();
        x &= ~(1 << 8);  // clear SPP to 0 for user mode
        x |= (1 << 5);   // enable interrupts in user mode
        w_sstatus(x);

        // set S Exception Program Counter to the saved user pc.
        w_sepc(p->trapframe->epc);
}

void usertrap(void) {
        // send interrupts and exceptions to kerneltrap(),
        // since we're now in the kernel.
        w_stvec((unsigned long)kernelvec);

        // acknowledge the software interrupt by clearing
        // the SSIP bit in sip.
        w_sip(r_sip() & ~2);

        usertrapret();
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
void w_sepc(unsigned long x) { asm volatile("csrw sepc, %0" : : "r"(x)); }

/*
| U |
| S | main -> procinit -> scheduler   kernelvec -> kerneltrap -> kernelvec -> scheduler      ...
|   |  ^                     |           ^                                        |           ^
|   |  |                     v           |                                        v           |
| M |_entry              timervec -------+                                    timervec -------+

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