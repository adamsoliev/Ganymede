#include "defs.h"

int main(void) {
        uartinit();

        while (1) {
                for (int i = 0; i < 100000; i++)
                        ;
                uartputc('h');
                uartputc('e');
                uartputc('l');
                uartputc('l');
                uartputc('o');
                uartputc('\n');
        }

        return 0;
}
