#include <stddef.h>
#include <stdint.h>

#define UART0 0x10000000L
#define Reg(reg) (volatile unsigned char *)(UART0 + reg)
#define ReadReg(reg) (*(Reg(reg)))
#define WriteReg(reg, v) (*(Reg(reg)) = (v))

void main();

// console output
void uartputc(int c) { WriteReg(0, c); }

// console input
int uartgetc(void) {
        // if (ReadReg(5) & 0x01) {
        return ReadReg(0);
        // }
        // return -1;
}

void print(const char *str) {
        while (*str != '\0') {
                uartputc(*str);
                str++;
        }
        return;
}

void main(void) {
        //
        print("Hello world!\r\n");
        while (1) {
                int c = uartgetc();
                uartputc(c);
        }
}
