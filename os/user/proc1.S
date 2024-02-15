.globl proc1
proc1:
        addi    sp,sp,-32
        sd      ra,24(sp)
        sd      s0,16(sp)
        addi    s0,sp,32
.L4:
        li      a4, 0
        li      a5,999424
        addi    a5,a5,575
        lui     t1,%hi(.LC0)
        addi    t0,t1,%lo(.LC0)
        j       .L2
.L3:
        addiw   a4,a4,1
.L2:
        ble     a4,a5,.L3
        add     a0, t0, zero
        # call    printf
        li      a7, 5 # assume printf
        ecall
        j       .L4

.LC0:
        .string "Process 1!\n"