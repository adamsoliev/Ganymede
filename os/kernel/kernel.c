#include "defs.h"

int main() {
        console_init();

        printf("--------------------------------\r\n");
        printf("<<      64-bit RISC-V OS      >>\r\n");
        printf("--------------------------------\r\n");

        char str[10];
        int read = console_read(str, 6);
        console_write(str, read);
        return 0;
}
