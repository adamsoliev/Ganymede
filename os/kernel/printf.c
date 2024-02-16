#include <stdarg.h>

#include "types.h"
#include "defs.h"

static char digits[] = "0123456789abcdef";

void printptr(uint64 num) {
        if (num == 0) {
                uartputc('(');
                uartputc('n');
                uartputc('i');
                uartputc('l');
                uartputc(')');
                return;
        }
        uartputc('0');
        uartputc('x');
        // print a char for every 4 bits, starting from MSB
        for (int i = 0; i < 16; i++, num <<= 4) {
                uartputc(digits[num >> 60]);
        }
}

// supports bases 10 and 16
void printint(int num, int base) {
        if (base != 10 && base != 16) return;

        char buf[16];
        int i = 0, sign = 0;
        unsigned int n;

        if ((sign = num < 0)) {
                n = -num;
        } else {
                n = num;
        }

        do {
                buf[i++] = digits[n % base];
        } while ((n /= base) != 0);

        if (sign) buf[i++] = '-';

        while (--i >= 0) {
                uartputc(buf[i]);
        }
}

// supports %d, %x, %s, %c, %p
void printf(char *format, ...) {
        if (format == 0) return;

        va_list argptr;
        va_start(argptr, format);

        for (int i = 0; format[i] != 0; i++) {
                char c = format[i];
                if (c != '%') {
                        uartputc(c);
                        continue;
                }
                c = format[++i];
                if (c == 'd') {
                        int x = va_arg(argptr, int);
                        printint(x, 10);
                } else if (c == 'x') {
                        uint64 x = va_arg(argptr, uint64);
                        printint(x, 16);
                } else if (c == 's') {
                        char *s = va_arg(argptr, char *);
                        if (s == 0) s = "empty string";
                        while (*s) {
                                uartputc(*s);
                                s++;
                        }
                } else if (c == 'c') {
                        c = (char)va_arg(argptr, int);
                        uartputc(c);
                } else if (c == 'p') {
                        uint64 p = va_arg(argptr, uint64);
                        printptr(p);
                } else {
                        uartputc('%');
                        if (c == '%') continue;
                        uartputc(c);
                }
        }
        va_end(argptr);
}