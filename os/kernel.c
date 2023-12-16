#include <assert.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

__attribute__((aligned(512))) char _stack[4096];

////////////////
// CONSOLE (& uart)
////////////////
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

////////////////
// PRINTS
////////////////
static char digits[] = "0123456789abcdef";
void printint(int xx, int base, int sign) {
        char buf[16];
        int i;
        unsigned int x;
        if (sign && (sign = xx < 0))
                x = -xx;
        else
                x = xx;

        i = 0;
        do {
                buf[i++] = digits[x % base];
        } while ((x /= base) != 0);

        if (sign) buf[i++] = '-';
        while (--i >= 0) uartputc(buf[i]);
}

void printptr(unsigned long x) {
        int i;
        uartputc('0');
        uartputc('x');
        for (i = 0; i < (int)(sizeof(unsigned long) * 2); i++, x <<= 4)
                uartputc(digits[x >> (sizeof(unsigned long) * 8 - 4)]);
}

void printf(char *fmt, ...) {
        va_list ap;
        int i, c;
        char *s;
        if (fmt == 0) {
                uartputc('n');
                uartputc('u');
                uartputc('l');
                uartputc('l');
        }

        va_start(ap, fmt);
        for (i = 0; (c = fmt[i] & 0xff) != 0; i++) {
                if (c != '%') {
                        uartputc(c);
                        continue;
                }
                c = fmt[++i] & 0xff;
                if (c == 0) break;
                switch (c) {
                        case 'd': printint(va_arg(ap, int), 10, 1); break;
                        case 'x': printint(va_arg(ap, int), 16, 1); break;
                        case 'p': printptr(va_arg(ap, unsigned long)); break;
                        case 's':
                                if ((s = va_arg(ap, char *)) == 0) s = "(null)";
                                for (; *s; s++) uartputc(*s);
                                break;
                        case '%': uartputc('%'); break;
                        default:
                                uartputc('%');
                                uartputc(c);
                                break;
                }
        }
        va_end(ap);
}

void panic(char *s) {
        printf("panic: ");
        printf(s);
        printf("\n");
}

////////////////
// RISC-V
////////////////
#define PGSIZE 4096  // bytes per page
#define PGSHIFT 12   // bits of offset within a page
#define PGROUNDUP(sz) (((sz) + PGSIZE - 1) & ~(PGSIZE - 1))
#define PGROUNDDOWN(a) (((a)) & ~(PGSIZE - 1))

#define PTE_V (1L << 0)  // valid
#define PTE_R (1L << 1)
#define PTE_W (1L << 2)
#define PTE_X (1L << 3)
#define PTE_U (1L << 4)  // user can access

// shift a physical address to the right place for a PTE.
#define PA2PTE(pa) ((((unsigned long)pa) >> 12) << 10)

#define PTE2PA(pte) (((pte) >> 10) << 12)

#define PTE_FLAGS(pte) ((pte) & 0x3FF)

// extract the three 9-bit page table indices from a virtual address.
#define PXMASK 0x1FF  // 9 bits
#define PXSHIFT(level) (PGSHIFT + (9 * (level)))
#define PX(level, va) ((((unsigned long)(va)) >> PXSHIFT(level)) & PXMASK)

// one beyond the highest possible virtual address.
// MAXVA is actually one bit less than the max allowed by
// Sv39, to avoid having to sign-extend virtual addresses
// that have the high bit set.
#define MAXVA (1L << (9 + 9 + 9 + 12 - 1))

////////////////
// ALLOCATOR
////////////////
#define KERNBASE 0x80000000L
#define PHYSTOP (KERNBASE + 128 * 1024 * 1024)  // 128 MB
#define TRAMPOLINE (MAXVA - PGSIZE)

// map kernel stacks beneath the trampoline,
// each surrounded by invalid guard pages.
#define KSTACK(p) (TRAMPOLINE - ((p) + 1) * 2 * PGSIZE)

char end[];  // first address after kernel
             // defined by virt.ld

struct run {
        struct run *next;
};

struct {
        struct run *freelist;
} kmem;

void kinit() {
        char *p = (char *)PGROUNDUP((unsigned long)end);
        for (; p + PGSIZE <= (char *)(void *)PHYSTOP; p += PGSIZE) {
                // add a page of physical memory pointed at by p to head of list
                struct run *r;
                r = (struct run *)p;
                r->next = kmem.freelist;
                kmem.freelist = r;
        }
}

// allocate one 4096-byte page of physical memory
void *kalloc(void) {
        struct run *r;
        r = kmem.freelist;
        if (r) kmem.freelist = r->next;
        return (void *)r;
}

void kfree(void *pa) {
        struct run *r;
        r = (struct run *)pa;
        r->next = kmem.freelist;
        kmem.freelist = r;
}

////////////////
// K PAGE TABLE
////////////////
typedef unsigned long *pagetable_t;

char etext[];  // virt.ld sets this to end of kernel code.

void *memset(void *dst, int c, unsigned int n);
unsigned long *walk(pagetable_t pagetable, unsigned long va, int alloc);
int mappages(pagetable_t pagetable, unsigned long va, unsigned long size, unsigned long pa,
             int perm);
void kvmmap(pagetable_t kpgtbl, unsigned long va, unsigned long pa, unsigned long sz, int perm);
pagetable_t kvmmake(void);
void proc_mapstacks(pagetable_t kpgtbl);

pagetable_t kernel_pagetable;

void kvminit(void) { kernel_pagetable = kvmmake(); }

// Make a direct-map page table for the kernel.
pagetable_t kvmmake(void) {
        pagetable_t kpgtbl;

        kpgtbl = (pagetable_t)kalloc();
        memset(kpgtbl, 0, PGSIZE);

        // uart registers
        kvmmap(kpgtbl, UART0, UART0, PGSIZE, PTE_R | PTE_W);

        // // virtio mmio disk interface
        // kvmmap(kpgtbl, VIRTIO0, VIRTIO0, PGSIZE, PTE_R | PTE_W);

        // PLIC
        // kvmmap(kpgtbl, PLIC, PLIC, 0x400000, PTE_R | PTE_W);

        // map kernel text executable and read-only.
        kvmmap(kpgtbl, KERNBASE, KERNBASE, (unsigned long)etext - KERNBASE, PTE_R | PTE_X);

        // map kernel data and the physical RAM we'll make use of.
        kvmmap(kpgtbl,
               (unsigned long)etext,
               (unsigned long)etext,
               PHYSTOP - (unsigned long)etext,
               PTE_R | PTE_W);

        // map the trampoline for trap entry/exit to
        // the highest virtual address in the kernel.
        // kvmmap(kpgtbl, TRAMPOLINE, (unsigned long)trampoline, PGSIZE, PTE_R | PTE_X);

        // allocate and map a kernel stack for each process.
        proc_mapstacks(kpgtbl);

        return kpgtbl;
}

void kvmmap(pagetable_t kpgtbl, unsigned long va, unsigned long pa, unsigned long sz, int perm) {
        if (mappages(kpgtbl, va, sz, pa, perm) != 0) panic("kvmmap");
}

int mappages(pagetable_t pagetable, unsigned long va, unsigned long size, unsigned long pa,
             int perm) {
        unsigned long a, last;
        unsigned long *pte;

        if (size == 0) panic("mappages: size");

        a = PGROUNDDOWN(va);
        last = PGROUNDDOWN(va + size - 1);
        for (;;) {
                if ((pte = walk(pagetable, a, 1)) == 0) return -1;
                if (*pte & PTE_V) panic("mappages: remap");
                *pte = PA2PTE(pa) | perm | PTE_V;
                if (a == last) break;
                a += PGSIZE;
                pa += PGSIZE;
        }
        return 0;
}

unsigned long *walk(pagetable_t pagetable, unsigned long va, int alloc) {
        if (va >= MAXVA) panic("walk");

        for (int level = 2; level > 0; level--) {
                unsigned long *pte = &pagetable[PX(level, va)];
                if (*pte & PTE_V) {
                        pagetable = (pagetable_t)PTE2PA(*pte);
                } else {
                        if (!alloc || (pagetable = (unsigned long *)kalloc()) == 0) return 0;
                        memset(pagetable, 0, PGSIZE);
                        *pte = PA2PTE(pagetable) | PTE_V;
                }
        }
        return &pagetable[PX(0, va)];
}

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
        //   struct spinlock lock;

        // p->lock must be held when using these:
        enum procstate state;  // Process state
        void *chan;            // If non-zero, sleeping on chan
        int killed;            // If non-zero, have been killed
        int xstate;            // Exit status to be returned to parent's wait
        int pid;               // Process ID

        // wait_lock must be held when using this:
        struct proc *parent;  // Parent process

        // these are private to the process, so p->lock need not be held.
        unsigned long kstack;         // Virtual address of kernel stack
        unsigned long sz;             // Size of process memory (bytes)
        pagetable_t pagetable;        // User page table
        struct trapframe *trapframe;  // data page for trampoline.S
        struct context context;       // swtch() here to run process
        struct file *ofile[16];       // Open files
        struct inode *cwd;            // Current directory
        char name[16];                // Process name (debugging)
};

#define NPROC 64  // maximum number of processes

struct proc proc[NPROC];

void proc_mapstacks(pagetable_t kpgtbl) {
        struct proc *p;

        for (p = proc; p < &proc[NPROC]; p++) {
                char *pa = kalloc();
                if (pa == 0) panic("kalloc");
                unsigned long va = KSTACK((int)(p - proc));
                kvmmap(kpgtbl, va, (unsigned long)pa, PGSIZE, PTE_R | PTE_W);
        }
}

////////////////
// TURN ON PAGING
////////////////
#define SATP_SV39 (8L << 60)
#define MAKE_SATP(pagetable) (SATP_SV39 | (((unsigned long)pagetable) >> 12))

void kvminithart();
static inline void w_satp(unsigned long x);
static inline void sfence_vma();

void kvminithart() {
        // wait for any previous writes to the page table memory to finish.
        sfence_vma();
        w_satp(MAKE_SATP(kernel_pagetable));
        // flush stale entries from the TLB.
        sfence_vma();
}

// flush all TLB entries
static inline void sfence_vma() { asm volatile("sfence.vma zero, zero"); }

static inline void w_satp(unsigned long x) { asm volatile("csrw satp, %0" : : "r"(x)); }

////////////////
// PROCESS TABLE
////////////////
void procinit(void) {
        struct proc *p;

        // initlock(&pid_lock, "nextpid");
        // initlock(&wait_lock, "wait_lock");
        for (p = proc; p < &proc[NPROC]; p++) {
                // initlock(&p->lock, "proc");
                p->state = UNUSED;
                p->kstack = KSTACK((int)(p - proc));
        }
}

void *memset(void *dst, int c, unsigned int n) {
        char *cdst = (char *)dst;
        for (unsigned int i = 0; i < n; i++) {
                cdst[i] = c;
        }
        return dst;
}

////////////////
// MAIN
////////////////
void main(void) {
        printf("%s", "------------------------------------\r\n");
        printf("%s", "<<<      64-bit RISC-V OS        >>>\r\n");
        printf("%s", "------------------------------------\r\n");

        uartinit();
        kinit();        // allocator
        kvminit();      // k page table
        kvminithart();  // turn on paging
        procinit();     // init process table

        unsigned long *page = kalloc();
        printf("allocated page: %p\r\n", page);
        kfree(page);
        unsigned long *page1 = kalloc();
        printf("allocated page1: %p\r\n", page1);

        while (1) {
                int c = uartgetc();
                if (c != -1) {
                        if (c == '\r')
                                uartputc('\n');
                        else if (c == 0x7f)
                                printf("%s", "\b \b");
                        else
                                uartputc(c);
                }
        }
}
