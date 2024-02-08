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