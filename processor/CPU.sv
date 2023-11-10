module CPU (
    input logic clk,       // system clock 
    input logic rst        // system reset
);

    logic [31:0] IMEM [0:4096];
    logic [63:0] DMEM [0:50];
    initial begin 
        $readmemh("./test/mem_instr", IMEM);
        $readmemh("./test/mem_data", DMEM);
    end

////////////////////////////////////////////
// IF
////////////////////////////////////////////

    logic [63:0] pc;
    wire [63:0] pc_next = pc + 4;
    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            pc <= 0;
        end else begin
            instr <= IMEM[pc];
            pc <= pc_next;
        end
    end

////////////////////////////////////////////
// ID
////////////////////////////////////////////
    logic [31:0] instr;

    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            instr <= 0;
        end else begin
            $display("%h", instr);
        end
    end

    wire [6:0] funct7 = instr[31:25];
    wire [4:0] rs2    = instr[24:20];
    wire [4:0] rs1    = instr[19:15];
    wire [2:0] funct3 = instr[14:12];
    wire [4:0] rd     = instr[11:7];
    wire [6:0] opcode = instr[6:0];

    wire [31:0] i_imm = { {20{instr[31]}}, instr[31:20] };
    wire [31:0] s_imm = { {20{instr[31]}}, instr[31:25], instr[11:7] };
    wire [31:0] b_imm = { {20{instr[31]}}, instr[7], instr[30:25], instr[11:8], 1'b0 };
    wire [31:0] u_imm = { instr[31:12], 12'b0 };
    wire [31:0] j_imm = { {11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0 };

////////////////////////////////////////////
// EX
////////////////////////////////////////////
    bit [63:0] registers [0:31];
    logic [63:0] id_rs1, id_rs2;
    assign id_rs1 = registers[rs1];
    assign id_rs2 = registers[rs2];

    always_comb begin 
        unique case (opcode)
            // LOAD
            7'b0000011: begin
            end
            // STORE
            7'b0100011: begin
            end
            // ALU
            default: ;
        endcase
    end

////////////////////////////////////////////
// MEM
////////////////////////////////////////////

////////////////////////////////////////////
// WB
////////////////////////////////////////////

endmodule
