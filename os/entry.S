.section .text
.global _entry
_entry:
        la 		sp, _stack

		li		t0, (0b11 << 11) | (1 << 7) | (1 << 3)
        csrw 	mstatus, t0

		la 		t1, main
        csrw 	mepc, t1

        csrw 	satp, zero

        mret