// Physical memory allocator, for user processes,
// kernel stacks, page-table pages,
// and pipe buffers. Allocates whole 4096-byte pages.

#include "defs.h"

#define KERNBASE 0x80000000L
#define PHYSTOP (KERNBASE + 1024 * 1024)

#define PGSIZE 4096  // bytes per page

#define PGROUNDUP(sz) (((sz) + PGSIZE - 1) & ~(PGSIZE - 1))
#define PGROUNDDOWN(a) (((a)) & ~(PGSIZE - 1))

void freerange(void *pa_start, void *pa_end);

extern char end[];  // first address after kernel.
                    // defined by kernel.ld.

struct run {
        struct run *next;
};

struct {
        struct run *freelist;
} kmem;

void kinit() { freerange(end, (void *)PHYSTOP); }

void freerange(void *pa_start, void *pa_end) {
        char *p;
        p = (char *)PGROUNDUP((unsigned long)pa_start);
        for (; p + PGSIZE <= (char *)pa_end; p += PGSIZE) {
                print("page\n");
                kfree(p);
        }
}

void kfree(void *pa) {
        print("kfree\n");
        struct run *r;

        if (((unsigned long)pa % PGSIZE) != 0 || (char *)pa < end || (unsigned long)pa >= PHYSTOP)
                panic("kfree");

        // Fill with junk to catch dangling refs.
        memset(pa, 1, PGSIZE);

        r = (struct run *)pa;

        r->next = kmem.freelist;
        kmem.freelist = r;
}

void *kalloc(void) {
        struct run *r;

        r = kmem.freelist;
        if (r) kmem.freelist = r->next;

        if (r) memset((char *)r, 5, PGSIZE);  // fill with junk
        return (void *)r;
}

void *memset(void *dst, int c, unsigned int n) {
        char *cdst = (char *)dst;
        int i;
        for (i = 0; i < n; i++) {
                cdst[i] = c;
        }
        return dst;
}