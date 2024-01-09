.section .text
.globl start      
start: 
        la a0, userStr
        li a7, 1
        ecall
        j start
        
userStr:
        .string "User 1\n"
        .zero 2
