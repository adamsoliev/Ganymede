#include "types.h"
#include "defs.h"

void kernelvec();

extern struct proc *cur_proc;

void kerneltrap() {
        print("kerneltrap\n");

        uint64 sstatus, scause, sepc;
        asm volatile("csrr %0, sstatus" : "=r"(sstatus));
        asm volatile("csrr %0, scause" : "=r"(scause));
        asm volatile("csrr %0, sepc" : "=r"(sepc));

        // acknowledge software interrupt
        asm volatile("csrc sip, %0" ::"r"(1 << 1));
        if (cur_proc != 0 && cur_proc->state == RUNNING) {
                intr_on();
                yield();
        }

        asm volatile("csrw sstatus, %0" ::"r"(sstatus));
        asm volatile("csrw sepc   , %0" ::"r"(sepc));
}

void trapinit() {
        // install kernel trap vec
        asm volatile("csrw stvec, %0" : : "r"(kernelvec));
}

void intr_on() {
        // enable S-mode interrupts
        asm volatile("csrs sstatus, %0" : : "r"(1 << 1));  // SIE
}

void timertrap() {
        print("timer interval\n");
        *(uint64 *)CLINT_MTIMECMP += INTERVAL;       // update mtimecmp
        asm volatile("csrs sip, %0" ::"r"(1 << 1));  // raise S-mode software interrupt
}