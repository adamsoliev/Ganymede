#include "types.h"
#include "defs.h"
#include "defines.h"

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
        p = (char *)PGROUNDUP((uint64)pa_start);
        for (; p + PGSIZE <= (char *)pa_end; p += PGSIZE) {
                kfree(p);
        }
}

void kfree(void *pa) {
        struct run *r;

        if (((uint64)pa % PGSIZE) != 0 || (char *)pa < end || (uint64)pa >= PHYSTOP) panic("kfree");

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
        for (int i = 0; i < n; i++) {
                cdst[i] = c;
        }
        return dst;
}

void *memmove(void *dst, const void *src, unsigned int n) {
        const char *s;
        char *d;

        if (n == 0) return dst;

        s = src;
        d = dst;
        if (s < d && s + n > d) {
                s += n;
                d += n;
                while (n-- > 0) *--d = *--s;
        } else
                while (n-- > 0) *d++ = *s++;

        return dst;
}