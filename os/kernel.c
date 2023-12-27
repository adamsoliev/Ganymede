// data structures
enum procstate { UNUSED, USED, SLEEPING, RUNNABLE, RUNNING, ZOMBIE };

struct proc {
        enum procstate state;
        int pid;
        unsigned long kstack;
        unsigned long sz;
        struct context context;
        char name[16];
};

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

//
#define NPROC 4
// since this OS runs on single CPU, we just have global vars for
struct proc *cur_proc;        // currently running process
struct context *cur_context;  // swtch() here to enter scheduler
struct proc proc[NPROC];      // process table

__attribute__((aligned(16))) char stack0[4096];

unsigned long timer_scratch[5];

void kernelvec();

void w_stvec(unsigned long x);
unsigned long r_sip();
void w_sip(unsigned long x);
unsigned long r_sstatus();
void w_sstatus(unsigned long x);

void scheduler(void);

// _entry jumps here
int main() {
        procinit();                         // init process table
        w_stvec((unsigned long)kernelvec);  // install kernel trap vector

        scheduler();
        return 0;
}

void scheduler(void) {
        for (;;) {
                //
                w_sstatus(r_sstatus() | (1 << 1));
                int a = 32 + 43;
        }
}

//////////////
// PROCESS
//////////////
void procinit(void) {
        struct proc *p;
        for (p = proc; p < &proc[NPROC]; p++) {
                p->state = UNUSED;
                p->kstack = 0;
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
