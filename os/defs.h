#include "defines.h"
#include "types.h"

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
void kvmmap(uint64 *ptable, uint64 va, uint64 pa, uint64 sz, int perm);

// proc.c
void procinit(void);
void allocproc(int pid);
void scheduler(void);
void yield(void);

// trap.c
void trapinit();
void intr_on();

// swtch.S
void swtch(struct context *old, struct context *new);

// printf.c
void printf(char *format, ...);