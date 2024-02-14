#include <stdarg.h>
#include <stdio.h>

#include "../types.h"

char digits[] = "0123456789abcdef";

void put(char c) {
        if (c == 0) return;
        putc(c, stdout);
}

void printptr(uint64 num) {
        if (num == 0) {
                put('(');
                put('n');
                put('i');
                put('l');
                put(')');
                return;
        }
        put('0');
        put('x');
        // print a char for every 4 bits, starting from MSB
        for (int i = 0; i < 16; i++, num <<= 4) {
                put(digits[num >> 60]);
        }
}

// supports bases 10 and 16
void printint(int num, int base) {
        if (base != 10 && base != 16) return;

        char buf[16];
        int i = 0, sign = 0;
        unsigned int n;

        if (sign = num < 0) {
                n = -num;
        } else {
                n = num;
        }

        do {
                buf[i++] = digits[n % base];
        } while ((n /= base) != 0);

        if (sign) buf[i++] = '-';

        while (--i >= 0) {
                put(buf[i]);
        }
}

// supports %d, %x, %s, %c, %p
void printf_custom(char *format, ...) {
        if (format == 0) return;

        va_list argptr;
        va_start(argptr, format);

        for (int i = 0; format[i] != 0; i++) {
                char c = format[i];
                if (c != '%') {
                        put(c);
                        continue;
                }
                c = format[++i];
                if (c == 'd' || c == 'x') {
                        uint64 x = va_arg(argptr, uint64);
                        printint(x, c == 'd' ? 10 : 16);
                } else if (c == 's') {
                        char *s = va_arg(argptr, char *);
                        if (s == 0) s = "empty string";
                        while (*s != '\0') put(*s++);
                } else if (c == 'c') {
                        c = (char)va_arg(argptr, int);
                        put(c);
                } else if (c == 'p') {
                        uint64 p = va_arg(argptr, uint64);
                        printptr(p);
                } else {
                        put('%');
                        if (c == '%') continue;
                        put(c);
                }
        }
        va_end(argptr);
}

int main() {
        printf_custom("%d\n", 29);
        printf("%d\n", 29);

        printf_custom("0x%x\n", 29);
        printf("0x%x\n", 29);

        printf_custom("%s\n", "hello");
        printf("%s\n", "hello");

        printf_custom("%%\n");
        printf("%%\n");

        printf_custom("%k\n");
        printf("%k\n");

        int num = 23;
        printf_custom("%c\n", num == 23 ? 'a' : 'b');
        printf("%c\n", num == 23 ? 'a' : 'b');

        printf_custom("%p\n", &num);
        printf("%p\n", &num);

        printf_custom("%d\n", 0);
        printf("%d\n", 0);

        printf_custom("%p\n", NULL);
        printf("%p\n", NULL);

        // int overflow
        printf_custom("%d\n", 9223372036854775807);
        printf("%d\n", 9223372036854775807);

        // int overflow
        printf_custom("%d\n", 4294967295);
        printf("%d\n", 4294967295);

        // int overflow by 1
        printf_custom("%d\n", 2147483648);
        printf("%d\n", 2147483648);

        // int underflow by 1
        printf_custom("%d\n", -2147483649);
        printf("%d\n", -2147483649);

        // int max
        printf_custom("%d\n", 2147483647);
        printf("%d\n", 2147483647);

        // int min
        printf_custom("%d\n", -2147483648);
        printf("%d\n", -2147483648);

        printf_custom("%d\n", 217867);
        printf("%d\n", 217867);

        printf_custom("%d\n", 277);
        printf("%d\n", 277);

        printf_custom("%d\n", 77);
        printf("%d\n", 77);

        return 0;
}
