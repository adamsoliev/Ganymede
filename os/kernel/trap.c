#include "types.h"
#include "defs.h"
#include "defines.h"
#include "proc.h"

void kernelvec();
extern char trampoline[], uservec[], userret[];

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
        asm volatile("csrs sstatus, %0" : : "r"(1 << 1));  // SIE
}

void intr_off() {
        asm volatile("csrc sstatus, %0" : : "r"(1 << 1));  // SIE
}

void timertrap() {
        print("timer interval\n");
        *(uint64 *)CLINT_MTIMECMP += INTERVAL;       // update mtimecmp
        asm volatile("csrs sip, %0" ::"r"(1 << 1));  // raise S-mode software interrupt
}

int copyin(uint64 *pagetable, char *dst, uint64 srcva, uint64 len) {
        uint64 n, va0, pa0;

        while (len > 0) {
                va0 = PGROUNDDOWN(srcva);
                pa0 = walkaddr(pagetable, va0);
                if (pa0 == 0) return -1;
                n = PGSIZE - (srcva - va0);
                if (n > len) n = len;
                memmove(dst, (void *)(pa0 + (srcva - va0)), n);

                len -= n;
                dst += n;
                srcva = va0 + PGSIZE;
        }
        return 0;
}

void usertrap() {
        // install kernelvec
        asm volatile("csrw stvec, %0" : : "r"(kernelvec));

        // save sepc
        uint64 sepc;
        asm volatile("csrr %0, sepc" : "=r"(sepc));
        cur_proc->trapframe->epc = sepc;

        uint64 scause;
        asm volatile("csrr %0, scause" : "=r"(scause));

        if (scause == 8) {  // syscall
                cur_proc->trapframe->epc += 4;
                intr_on();
                char str[11];
                copyin(cur_proc->pagetable, str, cur_proc->trapframe->a0, 11);
                printf("%s", str);
        } else if (scause & 1) {
                // acknowledge software interrupt
                asm volatile("csrc sip, %0" ::"r"(1 << 1));
                yield();
        }
        usertrapret();
}

extern uint64 *kptable;

void usertrapret() {
        intr_off();

        // install uservec (virtual address)
        uint64 trampoline_uservec = TRAMPOLINE + (uservec - trampoline);
        asm volatile("csrw stvec, %0" : : "r"(trampoline_uservec));

        // set up kernel info
        uint64 *ksatp;
        asm volatile("csrr %0, satp" : "=r"(ksatp));
        cur_proc->trapframe->kernel_satp = (uint64)ksatp;
        cur_proc->trapframe->kernel_sp = cur_proc->kstack + PGSIZE;
        cur_proc->trapframe->kernel_trap = (uint64)usertrap;

        // set prev mode to user and enable interrupts in that mode
        asm volatile("csrc sstatus, %0" ::"r"(1 << 8));
        asm volatile("csrs sstatus, %0" ::"r"(1 << 5));

        // user entry
        asm volatile("csrw sepc, %0" ::"r"(cur_proc->trapframe->epc));

        uint64 usatp = MAKE_SATP(cur_proc->pagetable);

        uint64 trampoline_userret = TRAMPOLINE + (userret - trampoline);
        ((void (*)(uint64))trampoline_userret)(usatp);
}