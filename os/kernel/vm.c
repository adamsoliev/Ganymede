#include "types.h"
#include "defs.h"

uint64 *walk(uint64 *ptable, uint64 va, int alloc);
void kvmmap(uint64 *ptable, uint64 va, uint64 pa, uint64 sz, int perm);

extern char etext[];

uint64 *kptable;

void kvminit() {
        kptable = kalloc();
        memset(kptable, 0, PGSIZE);

        // uart registers
        kvmmap(kptable, UART0, UART0, PGSIZE, PTE_R | PTE_W);
        // kernel code
        kvmmap(kptable, KERNBASE, KERNBASE, (uint64)etext - KERNBASE, PTE_R | PTE_X);
        // kernel data and rest of memory
        kvmmap(kptable, (uint64)etext, (uint64)etext, PHYSTOP - (uint64)etext, PTE_R | PTE_W);

        // turn on paging
        asm volatile("sfence.vma zero, zero");
        asm volatile("csrw satp, %0" : : "r"(MAKE_SATP(kptable)));
        asm volatile("sfence.vma zero, zero");
}

void kvmmap(uint64 *ptable, uint64 va, uint64 pa, uint64 sz, int perm) {
        uint64 *pte;
        if (sz == 0) panic("kvmmap: size\n");
        uint64 starta = PGROUNDDOWN(va);
        uint64 lasta = PGROUNDDOWN(va + sz - 1);
        for (;;) {
                if ((pte = walk(ptable, starta, 1)) == 0) panic("kvmap: walk\n");
                if (*pte & PTE_V) panic("kvmap: remap\n");
                *pte = PA2PTE(pa) | perm | PTE_V;
                if (starta == lasta) break;
                starta += PGSIZE;
                pa += PGSIZE;
        }
}

uint64 *walk(uint64 *ptable, uint64 va, int alloc) {
        if (va >= MAXVA) panic("walk\n");
        for (int level = 2; level > 0; level--) {
                uint64 *pte = &ptable[PX(level, va)];
                if (*pte & PTE_V)
                        ptable = (uint64 *)PTE2PA(*pte);
                else {
                        if (!alloc || (ptable = (uint64 *)kalloc()) == 0) return 0;
                        memset(ptable, 0, PGSIZE);
                        *pte = PA2PTE(ptable) | PTE_V;
                }
        }
        return &ptable[PX(0, va)];
}