	.globl	main
	.type	main,@function
main:
	addi	sp, sp, -32
	sd	ra, 24(sp)
	sd	s0, 16(sp)
	addi	s0, sp, 32
	li	a0, 0
	sw	a0, -20(s0)
	li	a0, -1
	ld	ra, 24(sp)
	ld	s0, 16(sp)
	addi	sp, sp, 32
	ret

