#include "defs.h"

__attribute__((aligned(128))) char stack0[4096];
int main(void);

void start() {
        // set prev to supervisor
        asm volatile("csrc mstatus, %0" ::"r"(0b11 << 11));
        asm volatile("csrs mstatus, %0" ::"r"(0b01 << 11));

        // supervisor entry
        asm volatile("csrw mepc, %0" ::"r"(main));

        // configure physical memory protection to give supervisor access to all memory
        asm volatile("csrw pmpaddr0, %0" ::"r"(0x3fffffffffffffULL));
        asm volatile("csrw pmpcfg0, %0" ::"r"(0xf));

        // timer interrupt
        /* 

                plan

                1. WITHOUT SOFTWARE INVOLVEMENT

                -- in start
                set mtimecmp
                set mstatus.MIE to enable M interrupts in general
                set mie.MTIE to enable M timer interrupts
                register a handler by writing 'mtvec'

                -- in the handler while trapped with timer interrupt
                save all regs
                print timer interrupt
                reset mtimecmp
                restore all regs
                mret

                2. INVOLVE SOFTWARE SO THAT PROCESSES CAN TAKE TURNS

                -- in start
                set mtimecmp
                set mstatus.MIE to enable M interrupts in general
                set mie.MTIE to enable M timer interrupts
                register a handler by writing 'mtvec'

                delegate S interrupts to S

                set sstatus.SIE to enable S interrupts in general
                set sie.STIE to enable S timer interrupts
                register a handler by writing 'stvec'

                -- in the handler while trapped with timer interrupt
                save some regs
                reset mtimecmp
                raise software timer interrupt via 'sip'
                restore some regs
                mret

                -- in S handler
                save all regs
                kerneltrap
                        save processor's state
                        read cause from 'scause' and call appropriate service routine
                        acknowledge software timer interrupt by clearing 'sip'
                        yield()
                        restore processor's state
                restore all regs
                sret

        ------------------------------------- GENERAL NOTES -------------------------------------
        mstatus 
                keeps track of and controls the hart’s current operating state
                sstatus - a restricted view of mstatus
                MIE & SIE  - global interrupt enable
                        lower-privilege mode interrupts are always off no matter 'wIE' bit for the lower-privilege mode
                        higher-privilege mode interrupts are always on no matter 'wIE' bit for the higher-privilege mode
                                higher-privilege-level code can use separate per-interrupt enable
                                bits to disable selected higher-privilege-mode interrupts 
                                before ceding control to a lower-privilege mode

                MPP, MPIE, MIE, MRET
                SPP, SPIE, SIE, SRET
        mtvec

        medeleg and mideleg
                by default, all traps at any privilege level are handled in machine mode; delegation can happen in two ways
                        through MRET
                        medeleg and mideleg

                setting a bit in medeleg or mideleg will delegate the corresponding trap, 
                when occurring in S-mode or U-mode, to the S-mode trap handler
                        when a trap occurs in this case,
                                scause -> trap cause 
                                sepc -> virtual address of the instruction that took the trap
                                stval -> exception-specific datum
                                SPP of mstatus -> active privilege mode at the time of the trap 
                                SPIE of mstatus -> value of the SIE field at the time of the trap
                                SIE of mstatus -> cleared.

                Traps never transition from a more-privileged mode to a less-privileged mode, but are allowed to be taken horizontally

                Delegated interrupts result in the interrupt being masked at the delegator privilege level
        
        mip and mie
                Restricted views of the mip and mie registers appear as the sip and sie registers for supervisor level

                mip.MEIP and mie.MEIE | mip.SEIP and mie.SEIE
                mip.MTIP and mie.MTIE | mip.STIP and mie.STIE
                mip.MSIP and mie.MSIE | mip.SSIP and mie.SSIE
        mepc
        mcause
        mtval

        mtime and mtimecmp
        ------------------------------------- END -------------------------------------

        */
        // asm volatile("csrs mie, %0" :: "r" (0b101 << 5)); // MTIE & STIE
        // asm volatile("csrs mstatus, %0" :: "r" (0b10)); // SIE
        // asm volatile("csrs mideleg, %0" :: "r" (0b1 << 5)); // STIE

        // switch to supervisor
        asm volatile("mret");
}

int main(void) {
        uartinit();

        uartputc('h');
        uartputc('e');
        uartputc('l');
        uartputc('l');
        uartputc('o');
        uartputc('\n');

        return 0;
}
