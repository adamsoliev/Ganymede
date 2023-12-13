.section .text
.global _entry
_entry:
        la 		sp, _stack

        # set M Previous Privilege mode to Supervisor, for mret
		li		t0, (0b11 << 11) | (1 << 7) | (1 << 3)
        csrw 	mstatus, t0

        # set M Exception Program Counter to main, for mret
		la 		t1, main
        csrw 	mepc, t1

        # delegate all interrupts and exceptions to S mode
        addi	a5,a5,-1 
        csrw	medeleg,a5
        csrw	mideleg,a5
        csrr	a5,sie
        ori	    a5,a5,546
        csrw	sie,a5

        # configure physical memory protection to S mode access to all physical memory
        li	    a5,-1
        srli	a5,a5,0xa
        csrw	pmpaddr0,a5
        li	    a5,15
        csrw	pmpcfg0,a5

        # disable paging for now
        csrw 	satp, zero

        # switch to supervisor mode and jump to main()
        mret
