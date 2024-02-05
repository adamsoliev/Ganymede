#include "defs.h"

// only > 128 works, not sure why
__attribute__ ((aligned (128))) char stack0[4096];

int console_read(char *buf, int len) {
        for (int i = 0; i < len; i++) {
                while (1) {
                        int c = uartgetc();
                        if (c != -1) {
                                buf[i] = c;
                                if (c == '\r') { /* Enter */
                                        buf[i] = 0;
                                        printf("\r\n");
                                        return i;
                                } else if (c == 0x03) { /* Ctrl + C */ 
                                        printf("Ctrl + C\r\n");
                                        return 0;
                                } else if (c == 0x7f) { /* Backspace */
                                        c = 0;
                                        if (i) printf("\b \b");
                                        i = i ? i - 2 : i - 1;
                                } else {
                                        printf("%c", c);
                                }
                                break;
                        }
                }
        }
        return len - 1;
}

void console_write(char *buf, int len) {
        for (int i = 0; i < len; i++) uartputc(buf[i]);
}

int main() {
        uartinit();

        printf("--------------------------------\r\n");
        printf("<<      64-bit RISC-V OS      >>\r\n");
        printf("--------------------------------\r\n");

        char str[10];
        int read = console_read(str, 6);
        console_write(str, read);
        return 0;
}
