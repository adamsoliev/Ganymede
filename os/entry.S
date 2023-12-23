.section .text
.global _entry
_entry:

    # clear 11 and 12 bits and set them to 01 (S-mode), for mret
    li      t0, 0x1800
    csrc    mstatus, t0
    li      t1, 0x800
    csrs    mstatus, t1

    # set mepc to C's main
    la		t2, main
    csrw	mepc, t2

    # disable paging 
    csrw	satp, zero

    # delegate all interrupts and exceptions to S-mode
    li      t4, 0xffff
    csrw    medeleg, t4
    li      t5, 0xffff
    csrw    mideleg, t5

    # return from M-mode to lower privilige mode (S in our case)
    mret

spin:
    j spin


# RISC-V PRIV 

# ---------------------------
# ecall      USER         ^
# --|---------------------|--
#   |                     |
#   v                   sret
#          SUPERVISOR
#  ecall                 ^
#   |                    |
# --|--------------------|---
#   v       MACHINE     mret
# ---------------------------

# mstatus
# SD MBE SBE SXL[1:0] UXL[1:0] TSR TW TVM MXR SUM MPRV XS[1:0] FS[1:0] MPP[1:0] VS[1:0] SPP MPIE UBE SPIE MIE SIE 
# 63 37  36  35-34    33-32    22  21 20  19  18  17   16-15    14-13    12-11    10-9   1    1   1   5    3   1

# mie
# MEIE SEIE MTIE STIE MSIE SSIE
# 11    9    7    5    3    1   

# mstatus       - M status     
# mepc          - M pc
# mie           - M interrupt enable
# medeleg       - M exception delegation
# mideleg       - M interrupt delegation
# mtvec         - M trap-vector base address
# mscratch      - M 
# mcounteren    - M counter enable

# sstatus   - S status
# sepc      - S pc
# sie       - S interrupt enable
# sip       - S interrupt pending
# stvec     - S trap-vector base address
# satp      - S address translation and protection
# scause    - S trap cause
# stval     - S trap value

# pmpcfg0   - physical memory protection
# pmpaddr0  - physical memory protection
