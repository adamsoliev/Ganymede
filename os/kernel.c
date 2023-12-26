__attribute__((aligned(16))) char stack0[4096];

unsigned long timer_scratch[5];

void kernelvec();

void w_stvec(unsigned long x);
unsigned long r_sip();
void w_sip(unsigned long x);
void scheduler(void);

// _entry jumps here
int main() {
        // install kernel trap vector
        w_stvec((unsigned long)kernelvec);

        scheduler();
        return 0;
}

void kerneltrap() {
        // acknowledge software interrupt by clearing sip.SSIP
        w_sip(r_sip() | (~2));
}

void scheduler(void) {
        for (;;) {
                //
                int a = 32 + 43;
        }
}

// RISC-V
void w_stvec(unsigned long x) { asm volatile("csrw stvec, %0" : : "r"(x)); }
unsigned long r_sip() {
        unsigned long x;
        asm volatile("csrr %0, sip" : "=r"(x));
        return x;
}
void w_sip(unsigned long x) { asm volatile("csrw sip, %0" : : "r"(x)); }
