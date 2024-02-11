#include "defines.h"

// main.c
void proc_mapstack(unsigned long *ptable);

// uart.c
void uartinit();
void uartputc(int c);
int uartgetc(void);
void print(char *c);
void panic(char *c);

// kalloc.c
void kinit();
void freerange(void *pa_start, void *pa_end);
void kfree(void *pa);
void *kalloc(void);
void *memset(void *dst, int c, unsigned int n);

// vm.c
void kvminit();
void kvmmap(unsigned long *ptable, unsigned long va, unsigned long pa, unsigned long sz, int perm);

// proc.c
void procinit(void);

// trap.c
void trapinit();
void intr_on();