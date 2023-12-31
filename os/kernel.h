
#define KERNBASE 0x80000000L
#define PHYSTOP (KERNBASE + 128 * 1024 * 1024)

#define PGSIZE 4096

#define NPROC 4

#define TRAMPOLINE (PHYSTOP - PGSIZE)
// #define KSTACK(p) (TRAMPOLINE - ((p) + 1) * 2 * PGSIZE)

#define TRAPFRAME (TRAMPOLINE - PGSIZE)

#define PGROUNDUP(sz) (((sz) + PGSIZE - 1) & ~(PGSIZE - 1))
#define PGROUNDDOWN(a) (((a)) & ~(PGSIZE - 1))
