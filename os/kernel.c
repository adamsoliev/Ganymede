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
void userret(void *);

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
unsigned long r_scause();

// string
char *safestrcpy(char *s, const char *t, int n);
void *memset(void *dst, int c, unsigned int n);

// os
int main();
void scheduler(void);
void swtch(struct context *, struct context *);

__attribute__((aligned(16))) char stack0[4096];

// process
struct proc proct[NPROC];
struct proc *cur_proc;
struct context cur_context;

// unsigned char initcode[] = {0x13, 0x03, 0x70, 0x01, 0x6f, 0xf0, 0xdf, 0xff};
// clang-format off
unsigned char initcode[] = {
        0x13, 0x07, 0x70, 0x01,
        0x17, 0x05, 0x00, 0x00, 0x13, 0x05, 0x45, 0x01, 0x93, 0x08, 0x10, 0x00, 0x73, 0x00, 0x00, 0x00, 
        0x6f, 0xf0, 0x1f, 0xff, 0x55, 0x73, 0x65, 0x72, 0x20, 0x31, 0x0a, 0x00, 0x00, 0x00, 0x01, 0x00
};
// clang-format on

// memory allocation
void *kalloc(void);
void kfree(void *pa);
void freerange(void *pa_start, void *pa_end);

// uart
void uartinit(void);
void print(const char *str);
void printptr(unsigned long x);
int uartgetc(void);
void uartputc(int c);

#define UART0 0x10000000L
#define Reg(reg) ((volatile unsigned char *)(UART0 + reg))

/* clang-format off */
// see https://www.lammertbies.nl/comm/info/serial-uart
#define RHR                     0               // receive holding register (for input bytes)
#define THR                     0               // transmit holding register (for output bytes)
#define IER                     1               // interrupt enable register
#define IER_RX_ENABLE           (1<<0)
#define IER_TX_ENABLE           (1<<1)
#define FCR                     2               // FIFO control register
#define FCR_FIFO_ENABLE         (1<<0)
#define FCR_FIFO_CLEAR          (3<<1)          // clear the content of the two FIFOs
#define ISR                     2               // interrupt status register
#define LCR                     3               // line control register
#define LCR_EIGHT_BITS          (3<<0)
#define LCR_BAUD_LATCH          (1<<7)          // special mode to set baud rate
#define LSR                     5               // line status register
#define LSR_RX_READY            (1<<0)          // input is waiting to be read from RHR
#define LSR_TX_IDLE             (1<<5)          // THR can accept another character to send

#define ReadReg(reg) (*(Reg(reg)))
#define WriteReg(reg, v) (*(Reg(reg)) = (v))
/* clang-format on */

extern char end[];  // first address after kernel.
                    // defined by kernel.ld.

///////////////////////////////////////////////////////////////////////////////
// _entry jumps here
int main() {
        uartinit();
        print("------------------------------------\r\n");
        print("<<<      64-bit RISC-V OS        >>>\r\n");
        print("------------------------------------\r\n");

        freerange(end, (void *)PHYSTOP);    // physical page allocator
        procinit();                         // process table
        w_stvec((unsigned long)kernelvec);  // kernel trap vector
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

void userinit(void) {
        struct proc *p = &proct[0];

        p->pid = 1;
        p->trapframe = (struct trapframe *)kalloc();
        print("trapframe:");
        printptr((unsigned long)p->trapframe);
        print("\n");

        // set up new context to start executing at usertrapret,
        // which returns to user space.
        p->context.ra = (unsigned long)usertrapret;
        p->context.sp = p->kstack + PGSIZE;  // stack grows downward, so point to top

        // copy initcode's instructions to mem starting at STARTADDR
        // pc has to be set to that address too
        char *STARTADDR = kalloc();  // for user code, data and stack
        print("STARTADDR:");
        printptr((unsigned long)STARTADDR);
        print("\n");
        char *copy = STARTADDR;
        for (int i = 0; i < (int)sizeof(initcode); i++) {
                *copy = initcode[i];
                copy++;
        }
        p->sz = PGSIZE;

        // prepare for the very first 'return' from kernel to user
        p->trapframe->epc = (unsigned long)STARTADDR;          // user pc (bottom)
        p->trapframe->sp = (unsigned long)STARTADDR + PGSIZE;  // user stack pointer (top)

        safestrcpy(p->name, "initcode", sizeof(p->name));

        p->state = RUNNABLE;
}

void procinit(void) {
        struct proc *p;
        for (p = proct; p < &proct[NPROC]; p++) {
                p->state = UNALLOCATED;
                p->kstack = (unsigned long)kalloc();
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

        if (r_scause() == 8) {
                // syscall
                p->trapframe->epc += 4;
                w_sstatus(r_sstatus() | (1 << 1));
                // int num = p->trapframe->a7;           // 1 is print
                unsigned long a0 = p->trapframe->a0;  // str address for printing
                print((const char *)a0);
        } else {
                // acknowledge the software interrupt by clearing
                // the SSIP bit in sip.
                w_sip(r_sip() & ~2);
        }

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
        userret(p->trapframe);
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
unsigned long r_scause() {
        unsigned long x;
        asm volatile("csrr %0, scause" : "=r"(x));
        return x;
}

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

void *memset(void *dst, int c, unsigned int n) {
        char *cdst = (char *)dst;
        unsigned int i;
        for (i = 0; i < n; i++) {
                cdst[i] = c;
        }
        return dst;
}

////////////////////////////////
// MEMORY ALLOCATION
////////////////////////////////
struct run {
        struct run *next;
};

struct {
        struct run *freelist;
} kmem;

void freerange(void *pa_start, void *pa_end) {
        char *p;
        p = (char *)PGROUNDUP((unsigned long)pa_start);
        for (; p + PGSIZE < (char *)pa_end; p += PGSIZE) kfree(p);
}

// Free the page of physical memory pointed at by pa,
// which normally should have been returned by a
// call to kalloc().  (The exception is when
// initializing the allocator; see kinit above.)
void kfree(void *pa) {
        struct run *r;

        // if (((unsigned long)pa % PGSIZE) != 0 || (char *)pa < end || (unsigned long)pa >= PHYSTOP)
        //         panic("kfree");

        // Fill with junk to catch dangling refs.
        memset(pa, 1, PGSIZE);

        r = (struct run *)pa;

        r->next = kmem.freelist;
        kmem.freelist = r;
}

// Allocate one 4096-byte page of physical memory.
// Returns a pointer that the kernel can use.
// Returns 0 if the memory cannot be allocated.
void *kalloc(void) {
        struct run *r;

        r = kmem.freelist;
        if (r) kmem.freelist = r->next;

        if (r) memset((char *)r, 5, PGSIZE);  // fill with junk
        return (void *)r;
}

// uart
void uartinit(void) {
        WriteReg(IER, 0x00);            // disable interrupts.
        WriteReg(LCR, LCR_BAUD_LATCH);  // special mode to set baud rate.
        WriteReg(0, 0x03);              // LSB for baud rate of 38.4K.
        WriteReg(1, 0x00);              // MSB for baud rate of 38.4K.
        WriteReg(LCR, LCR_EIGHT_BITS);  // and set word length to 8 bits, no parity.
        WriteReg(FCR, FCR_FIFO_ENABLE | FCR_FIFO_CLEAR);  // reset and enable FIFOs.
        WriteReg(IER, IER_TX_ENABLE | IER_RX_ENABLE);     // enable transmit and receive interrupts.
}

void uartputc(int c) { WriteReg(0, c); }

int uartgetc(void) {
        if (ReadReg(5) & 0x01) return ReadReg(0);
        return -1;
}

void print(const char *str) {
        while (*str != '\0') {
                uartputc(*str);
                str++;
        }
        return;
}

static char digits[] = "0123456789abcdef";

void printptr(unsigned long x) {
        int i;
        uartputc('0');
        uartputc('x');
        for (i = 0; i < (sizeof(unsigned long) * 2); i++, x <<= 4)
                uartputc(digits[x >> (sizeof(unsigned long) * 8 - 4)]);
}
