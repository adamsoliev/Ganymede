.section .text
.global _entry
_entry:
        la 		sp, stack0

        // set prev to supervisor mode
        csrr        t0, mstatus
        li          t1, (0b11 << 11)
        not         t1, t1              // mask
        and         t0, t0, t1          // clear 11th & 12th
		li		    t1, (0b1 << 11)     // set 11th & 12th 
        or          t0, t0, t1
        csrw 	    mstatus, t0

		la 		    t1, main
        csrw 	    mepc, t1

        // turn off virt mem
        csrw 	    satp, zero

        // machine (exception, interrupt) delegation to supervisor 
        lui         t0, 0x10 
        addi        t0, t0, -1          // 0xffff
        csrw        medeleg, t0
        csrw        mideleg, t0
        csrr        t0, sie

        // external | timer | software
        li          t0, (0b1 << 9) | (0b1 << 5) | (0b1 << 1) 
        or          t0, t0, t1
        csrw        sie, t0             // enable corresponding (exception, interrupt) in supervisor

        // configure PMP to give supervisor mode 
        // access to all pmem
        li          t0, -1
        srli        t0, t0, 0xa
        csrw        pmpaddr0, t0
        li          t0, 15
        csrw        pmpcfg0, t0

        mret


# When a trap is delegated to S-mode
# scause, sepc, stval, SPP field of mstatus, SPIE field of mstatus, SIE field of mstatus