#include "defs.h"

int main() {
        uartinit();

        int c = uartgetc();
        uartputc(c);
        uartputc('H');
        uartputc('E');
        uartputc('L');
        uartputc('L');
        uartputc('O');
        return 0;
}
