#include "defs.h"

int main(void) {
        uartinit();

        while (1) {
                for (int i = 0; i < 100000000; i++)
                        ;
                print("HELLO WORLD!\n");
        }

        return 0;
}
