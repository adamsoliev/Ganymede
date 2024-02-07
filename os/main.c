#include "defs.h"

char s[6] = "hello\n";

int main(void) {
        console_init();
        // uartinit();

        while (1) {
                for (int i = 0; i < 1000000; i++)
                        ;

                printf(s);
                // uartputc('h');
                // uartputc('e');
                // uartputc('l');
                // uartputc('l');
                // uartputc('o');
                // uartputc('\n');
        }

        return 0;
}
