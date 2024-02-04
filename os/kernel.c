#include "defs.h"

void printf(const char *str) {
        while (*str != '\0') {
                uartputc(*str);
                str++;
        }
        return;
}

int main() {
        uartinit();

        printf("--------------------------------\r\n");
        printf("<<      64-bit RISC-V OS      >>\r\n");
        printf("--------------------------------\r\n");

        while (1) {
                int c = uartgetc();
                if (c != -1) {
                        if (c == '\r') uartputc('\n');
                        else if (c == 0x7f) {
                                uartputc('\b');
                                uartputc(' ');
                                uartputc('\b');
                        }
                        else uartputc(c);
                }
        }
        return 0;
}
