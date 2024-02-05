.section .text
.global _entry
_entry:
        la 		sp, stack0

        // set prev to supervisor mode
		li		t0, (0b1 << 11)
        csrw 	mstatus, t0

		la 		t1, main
        csrw 	mepc, t1

        // configure PMP to give supervisor mode 
        // access to all pmem
        li      t0, -1
        srli    t0, t0, 0xa
        csrw    pmpaddr0, t0
        li      t0, 15
        csrw    pmpcfg0, t0

        // turn off virt mem
        csrw 	satp, zero

        mret
