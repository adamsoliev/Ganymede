#include "defs.h"

// only > 128 works, not sure why
__attribute__ ((aligned (128))) char stack0[4096];

int main() {
        uartinit();

        printf("--------------------------------\r\n");
        printf("<<      64-bit RISC-V OS      >>\r\n");
        printf("--------------------------------\r\n");

        while (1) {
                int c = uartgetc();
                if (c != -1) {
                        if (c == '\r')
                                printf("%c", '\n');
                        else if (c == 0x7f) {
                                printf("\b");
                                printf(" ");
                                printf("\b");
                        } else
                                printf("%c", c);
                }
        }
        return 0;
}
