//
// formatted console output -- printf, panic.
//
#include <stdarg.h>

#include "defs.h"

volatile int panicked = 0;

static char digits[] = "0123456789abcdef";

static void printint(int xx, int base, int sign) {
        char buf[16];
        int i;
        unsigned int x;

        if (sign && (sign = xx < 0))
                x = -xx;
        else
                x = xx;

        i = 0;
        do {
                buf[i++] = digits[x % base];
        } while ((x /= base) != 0);

        if (sign) buf[i++] = '-';

        while (--i >= 0) uartputc(buf[i]);
}

static void printptr(unsigned long x) {
        int i;
        uartputc('0');
        uartputc('x');
        for (i = 0; i < (sizeof(unsigned long) * 2); i++, x <<= 4)
                uartputc(digits[x >> (sizeof(unsigned long) * 8 - 4)]);
}

// Print to the console. only understands %d, %x, %p, %s
void printf(char *fmt, ...) {
        va_list ap;
        int i, c;
        char *s;

        if (fmt == 0) panic("null fmt");

        va_start(ap, fmt);
        for (i = 0; (c = fmt[i] & 0xff) != 0; i++) {
                if (c != '%') {
                        uartputc(c);
                        continue;
                }
                c = fmt[++i] & 0xff;
                if (c == 0) break;
                switch (c) {
                        case 'd': printint(va_arg(ap, int), 10, 1); break;
                        case 'x': printint(va_arg(ap, int), 16, 1); break;
                        case 'p': printptr(va_arg(ap, unsigned long)); break;
                        case 's':
                                if ((s = va_arg(ap, char *)) == 0) s = "(null)";
                                for (; *s; s++) uartputc(*s);
                                break;
                        // case 'c': uartputc(va_arg(ap, int)); break;
                        case '%': uartputc('%'); break;
                        default:
                                // Print unknown % sequence to draw attention.
                                uartputc('%');
                                uartputc(c);
                                break;
                }
        }
        va_end(ap);
}

void panic(char *s) {
        printf("panic: ");
        printf(s);
        printf("\n");
        panicked = 1;  // freeze uart output from other CPUs
        for (;;)
                ;
}
