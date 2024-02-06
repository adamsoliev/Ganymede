#define CLINT 0x2000000L
#define CLINT_MTIMECMP (CLINT + 0x4000)
#define CLINT_MTIME (CLINT + 0xBFF8) // cycles since boot.

// uart.c
void uartinit(void);
int uartgetc(void);
void uartputc(int c);

// printf.c
void printf(char *fmt, ...);
void panic(char *s);

// console.c
int console_read(char *buf, int len);
void console_write(char *buf, int len);
void console_init();