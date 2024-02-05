
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