#include <stddef.h>
#include <stdint.h>

__attribute__((aligned(512))) char _stack[4096];

#define UART0 0x10000000L
#define Reg(reg) ((volatile unsigned char *)(UART0 + reg))

/* clang-format off */
// see https://www.lammertbies.nl/comm/info/serial-uart
#define RHR                     0               // receive holding register (for input bytes)
#define THR                     0               // transmit holding register (for output bytes)
#define IER                     1               // interrupt enable register
#define IER_RX_ENABLE           (1<<0)
#define IER_TX_ENABLE           (1<<1)
#define FCR                     2               // FIFO control register
#define FCR_FIFO_ENABLE         (1<<0)
#define FCR_FIFO_CLEAR          (3<<1)          // clear the content of the two FIFOs
#define ISR                     2               // interrupt status register
#define LCR                     3               // line control register
#define LCR_EIGHT_BITS          (3<<0)
#define LCR_BAUD_LATCH          (1<<7)          // special mode to set baud rate
#define LSR                     5               // line status register
#define LSR_RX_READY            (1<<0)          // input is waiting to be read from RHR
#define LSR_TX_IDLE             (1<<5)          // THR can accept another character to send

#define ReadReg(reg) (*(Reg(reg)))
#define WriteReg(reg, v) (*(Reg(reg)) = (v))
/* clang-format on */

void main();

void uartinit(void) {
        WriteReg(IER, 0x00);            // disable interrupts.
        WriteReg(LCR, LCR_BAUD_LATCH);  // special mode to set baud rate.
        WriteReg(0, 0x03);              // LSB for baud rate of 38.4K.
        WriteReg(1, 0x00);              // MSB for baud rate of 38.4K.
        WriteReg(LCR, LCR_EIGHT_BITS);  // and set word length to 8 bits, no parity.
        WriteReg(FCR, FCR_FIFO_ENABLE | FCR_FIFO_CLEAR);  // reset and enable FIFOs.
        WriteReg(IER, IER_TX_ENABLE | IER_RX_ENABLE);     // enable transmit and receive interrupts.
}

void uartputc(int c) { WriteReg(0, c); }

int uartgetc(void) {
        if (ReadReg(5) & 0x01) return ReadReg(0);
        return -1;
}

void print(const char *str) {
        while (*str != '\0') {
                uartputc(*str);
                str++;
        }
        return;
}

void main(void) {
        uartinit();
        print("------------------------------------\r\n");
        print("<<<      64-bit RISC-V OS        >>>\r\n");
        print("------------------------------------\r\n");

        while (1) {
                int c = uartgetc();
                if (c != -1) {
                        if (c == '\r')
                                uartputc('\n');
                        else if (c == 0x7f)
                                print("\b \b");
                        else
                                uartputc(c);
                }
        }
}

/*
Address spaces: kernel and per-process

*/
