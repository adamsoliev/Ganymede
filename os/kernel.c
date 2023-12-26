__attribute__((aligned(16))) char stack0[4096];

unsigned long timer_scratch[5];

void scheduler(void);

// _entry jumps here
int main() {
        // initialize subsystems
        scheduler();
        return 0;
}

void scheduler(void) {
        for (;;) {
                //
        }
}
