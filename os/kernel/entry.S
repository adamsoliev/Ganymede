.section .text
.global _entry
_entry:
    // set up stack and jump to start in start.c
    la sp, stack0
    li a0, 4096
    add sp, sp, a0
    call start 

spin:
    j spin