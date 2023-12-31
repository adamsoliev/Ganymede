// constants
#define KERNBASE 0x80000000L
#define PHYSTOP (KERNBASE + 128 * 1024 * 1024)

#define PGSIZE 4096

#define NPROC 4

#define TRAMPOLINE (PHYSTOP - PGSIZE)  // qemu's physical address space
#define KSTACK(p) (TRAMPOLINE - ((p) + 1) * 2 * PGSIZE)

// trap
void kernelvec();

unsigned long timer_scratch[5];

// riscv
void w_stvec(unsigned long x);
unsigned long r_sip();
void w_sip(unsigned long x);
unsigned long r_sstatus();
void w_sstatus(unsigned long x);

// os
int main();
void scheduler(void);

__attribute__((aligned(16))) char stack0[4096];


// data structures
enum procstate { UNUSED, USED, RUNNABLE, RUNNING };

struct proc {
        enum procstate state;
        int pid;

        unsigned long kstack;
};
// process
struct proc proct[NPROC];

///////////////////////////////////////////////////////////////////////////////
// _entry jumps here
int main() {
        procinit();                         // process table
        w_stvec((unsigned long)kernelvec);  // install kernel trap vector

        scheduler();
        return 0;
}

void scheduler(void) {
        struct proc *p;
        for (;;) {
                w_sstatus(r_sstatus() | (1 << 1));
                for (p = proct; p < &proct[NPROC]; p++) {
                        if (p->state == RUNNABLE) {
                                p->state = RUNNING;
                                // cur_proc = p
                                // swtch
                        }
                }
        }
}

// process
// unsigned char initcode[] = {0x13, 0x03, 0x70, 0x01, 0x6f, 0xf0, 0xdf, 0xff};

void procinit(void) {
        struct proc *p;
        for (p = proct; p < &proct[NPROC]; p++) {
                p->state = UNUSED;
                p->kstack = KSTACK((int)(p - proct));
        }
}

// trap
void kerneltrap() {
        // acknowledge software interrupt by clearing sip.SSIP
        w_sip(r_sip() & (~2));
}

// risc-v
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
0x80000000 - _entry
0x800000b4 - spin
0x800000c0 - timervec
0x800000f0 - kernelvec
0x80000182 - main
0x800001a8 - kerneltrap
0x800001c8 - scheduler
0x80001140 - initcode
0x80001150 - stack0
*/