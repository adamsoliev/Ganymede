
## Design

![64-bit RISC-V Core design](./assets/RISCV_25_10_23.png)

5-stage pipelined 64-bit RISC-V soft core
- Supported instructions: add, sub, sll, slt, sltu, xor, or, and, bne, lui, auipc, jal, lb, lbu, lh, lhu, lw, lwu, ld, sb, sh, sw, sd, addi, slli, slti, sltiu, xori, ori, andi, addiw, slliw, addw, subw, sllw
- Forwarding for RAW hazards
    - MEM   -> EX
    - WB    -> EX
    - WB    -> ID (implicitly through Register File)
- Stalling for load-use hazards
- Static 'not taken' branch predictor

