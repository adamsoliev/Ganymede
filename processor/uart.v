`default_nettype none

module uart(
    input wire clk,
    input wire wr,
    input wire [7:0] data,
    output wire tx,
    output reg busy
);

	parameter [23:0] CLOCKS_PER_BAUD = 24'd868;
	localparam [3:0] START	= 4'h0,
		BIT_ZERO	= 4'h1,
		BIT_ONE		= 4'h2,
		BIT_TWO		= 4'h3,
		BIT_THREE	= 4'h4,
		BIT_FOUR	= 4'h5,
		BIT_FIVE	= 4'h6,
		BIT_SIX		= 4'h7,
		BIT_SEVEN	= 4'h8,
		LAST		= 4'h8,
		IDLE		= 4'hf;

	reg	[23:0]	counter;
	reg	[3:0]	state;
    reg [8:0] l_data;
	reg	baud_stb;

    initial	busy = 1'b0;
	initial	state  = IDLE;
	always @(posedge clk) begin
        if ((wr)&&(!busy)) begin
            { busy, state } <= { 1'b1, START };
        end
        else if (baud_stb) begin
            if (state == IDLE) begin
                { busy, state } <= { 1'b0, IDLE };
            end
            else if (state < LAST) begin
                busy <= 1'b1;
                state <= state + 1'b1;
            end 
            else begin
                { busy, state } <= { 1'b1, IDLE };
            end
        end
    end

    initial	l_data = 9'h1ff;
	always @(posedge clk) begin
        if ((wr)&&(!busy)) begin
            l_data <= { data, 1'b0 };
        end
        else if (baud_stb) begin
            l_data <= { 1'b1, l_data[8:1] };
        end
    end
    assign tx = l_data[0];

    initial	baud_stb = 1'b1;
	initial	counter = 0;
	always @(posedge clk) begin
        if ((wr)&&(!busy)) begin
            counter  <= CLOCKS_PER_BAUD - 1'b1;
            baud_stb <= 1'b0;
        end else if (!baud_stb) begin
            baud_stb <= (counter == 24'h01);
            counter  <= counter - 1'b1;
        end else if (state != IDLE) begin
            counter <= CLOCKS_PER_BAUD - 1'b1;
            baud_stb <= 1'b0;
        end
    end

endmodule
