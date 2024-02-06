// https://www.lammertbies.nl/comm/info/serial-uart
#define UART0 0x10000000L
#define IER 1  // interrupt enable
#define FCR 2  // fifo control
#define LCR 3  // line control
#define LSR 5  // line status

#define Reg(reg) ((volatile unsigned char *)(UART0 + reg))

void uartinit() {
        (*(Reg(IER))) = 0x00;      // turn off interrupts
        (*(Reg(LCR))) = (1 << 7);  // access rate setter registers
        (*(Reg(0))) = 0x01;        // DLL
        (*(Reg(1))) = 0x00;        // DLM
        (*(Reg(LCR))) = 0x11;      // data size 8-bit
        (*(Reg(FCR))) = (unsigned char) 0x111;     // clear Tx FIFO | clear Rx FIFO | Enable FIFO
        (*(Reg(IER))) = 0x11;      // enable Tx | enable Rx
}

void uartputc(int c) {
        while (((*(Reg(LSR))) & (1 << 5)) == 0)
                ;
        *(Reg(0)) = c;
}

int uartgetc(void) {
        if ((*(Reg(LSR))) & 0x1) {
                return *(Reg(0));
        } else
                return -1;
}