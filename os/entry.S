

# Define a .text.init section.
.section .text.init

# Execution starts here.
.global _start
_start:
	la		sp, _stack
	# Set 'mstatus' register
	li		t0, (0b11 << 11) | (1 << 7) | (1 << 3)
	csrw	mstatus, t0
	# Set 'mepc' register
	la		t1, main
	csrw	mepc, t1
	mret


    
