.section .text
.globl start      
start: 
        li a4, 23
        la a0, userStr
        li a7, 1
        ecall
        j start
        
userStr:
        .string "User 1\n"
        .zero 2
