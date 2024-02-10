#include "defs.h"

/*
PAGING
 - allocate kernel page table
 - turn on paging
*/

unsigned long *walk(unsigned long *ptable, unsigned long va, int alloc);
void kvmmap(unsigned long *ptable, unsigned long va, unsigned long pa, unsigned long sz, int perm);

extern char etext[];

unsigned long *kptable;

void kvminit() {
        kptable = kalloc();
        memset(kptable, 0, PGSIZE);

        // uart registers
        kvmmap(kptable, UART0, UART0, PGSIZE, PTE_R | PTE_W);
        // kernel code
        kvmmap(kptable, KERNBASE, KERNBASE, (unsigned long)etext - KERNBASE, PTE_R | PTE_X);
        // kernel data and rest of memory
        kvmmap(kptable,
               (unsigned long)etext,
               (unsigned long)etext,
               PHYSTOP - (unsigned long)etext,
               PTE_R | PTE_W);

        // turn on paging
        asm volatile("sfence.vma zero, zero");
        asm volatile("csrw satp, %0" : : "r"(MAKE_SATP(kptable)));
        asm volatile("sfence.vma zero, zero");
}

void kvmmap(unsigned long *ptable, unsigned long va, unsigned long pa, unsigned long sz, int perm) {
        unsigned long *pte;
        if (sz == 0) panic("kvmmap: size\n");
        unsigned int starta = PGROUNDDOWN(va);
        unsigned int lasta = PGROUNDDOWN(va + sz - 1);
        for (;;) {
                if ((pte = walk(ptable, starta, 1)) == 0) panic("kvmap: walk\n");
                if (*pte & PTE_V) panic("kvmap: remap\n");
                *pte = PA2PTE(pa) | perm | PTE_V;
                if (starta == lasta) break;
                starta += PGSIZE;
                pa += PGSIZE;
        }
}

unsigned long *walk(unsigned long *ptable, unsigned long va, int alloc) {
        if (va >= MAXVA) panic("walk\n");
        for (int level = 2; level > 0; level--) {
                unsigned long *pte = &ptable[PX(level, va)];
                if (*pte & PTE_V)
                        ptable = (unsigned long *)PTE2PA(*pte);
                else {
                        if (!alloc || (ptable = (unsigned long *)kalloc()) == 0) return 0;
                        memset(ptable, 0, PGSIZE);
                        *pte = PA2PTE(ptable) | PTE_V;
                }
        }
        return &ptable[PX(0, va)];
}

int main(void) {
        uartinit();
        kinit();
        kvminit();

        while (1) {
                for (int i = 0; i < 100000000; i++)
                        ;
                print("HELLO WORLD!\n");
        }

        return 0;
}