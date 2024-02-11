#define UART0 0x10000000L

// core local interruptor (CLINT), which contains the timer
#define CLINT 0x2000000L
#define CLINT_MTIMECMP (CLINT + 0x4000)
#define CLINT_MTIME (CLINT + 0xBFF8)  // cycles since boot

#define INTERVAL 10000000

#define KERNBASE 0x80000000L
#define PHYSTOP (KERNBASE + 17 * 1024 * 1024)

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

#define MAXVA (1L << (9 + 9 + 9 + 12 - 1))

// use riscv's sv39 page table scheme.
#define SATP_SV39 (8L << 60)
#define MAKE_SATP(pagetable) (SATP_SV39 | (((unsigned long)pagetable) >> 12))

// map kernel stacks beneath the trampoline,
// each surrounded by invalid guard pages.
#define KSTACK(p) ((MAXVA - PGSIZE) - ((p) + 1) * 2 * PGSIZE)
