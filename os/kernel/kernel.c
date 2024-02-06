#include "defs.h"

void kernelvec();

int main() {
        console_init();

        printf("--------------------------------\r\n");
        printf("<<      64-bit RISC-V OS      >>\r\n");
        printf("--------------------------------\r\n");

        asm volatile("csrw stvec, %0" :: "r"((unsigned long)kernelvec));

        while (1) {
                printf("mtimecmp: %d\n", *(unsigned long*)CLINT_MTIMECMP);
                printf("mtime   : %d\n", *(unsigned long*)CLINT_MTIME);
                // char str[10];
                // int read = console_read(str, 6);
                // console_write(str, read);
        }
        return 0;

}

void kerneltrap() {
        printf("kerneltrap\n");
}
