
.globl write
write:
    li a7, 1
    ecall 
    ret

.globl read
read:
    li a7, 2
    ecall 
    ret