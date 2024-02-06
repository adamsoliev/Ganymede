#include "defs.h"

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
                                        // printf("%c", c);
                                        uartputc(c);
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

void console_init() {
        uartinit();
}