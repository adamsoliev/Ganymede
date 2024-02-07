
// uart.c
void uartinit();
void uartputc(int c);
int uartgetc(void);

// console.c
void console_init();
void console_write(char *buf, int len);
int console_read(char *buf, int len);
void printf(char *fmt, ...);