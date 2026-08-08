module TPU(
    clk,
    rst_n,

    in_valid,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out
);

input clk;
input rst_n;
input            in_valid;
input [7:0]      K;
input [7:0]      M;
input [7:0]      N;
output  reg      busy;

output           A_wr_en;
output [15:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output [15:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output  reg          C_wr_en;
output  reg [15:0]   C_index;
output  reg [127:0]  C_data_in;
input  [127:0]       C_data_out;

//====================================================================
// TPU 只讀 A/B，不寫入
//====================================================================
assign A_wr_en   = 1'b0;
assign A_data_in = 32'd0;
assign B_wr_en   = 1'b0;
assign B_data_in = 32'd0;

//====================================================================
// global_buffer 假設為組合邏輯讀（同拍位址同拍資料）。
// 若實際 global_buffer.v 是同步暫存輸出（延遲1拍），
// 麻煩告知，我再調整 pipeline。
//====================================================================
localparam MARGIN = 8;   // 3(row skew) + 3(col skew) + 緩衝

//====================================================================
// FSM
//====================================================================
localparam S_IDLE      = 3'd0,
           S_INIT_TILE = 3'd1,
           S_FEED      = 3'd2,
           S_WRITE     = 3'd3,
           S_NEXT_TILE = 3'd4,
           S_DONE      = 3'd5;

reg [2:0]  state;

reg [7:0]  K_r, M_r, N_r;
reg [7:0]  m_tiles, n_tiles;
reg [7:0]  m_tile, n_tile;

reg [8:0]  cyc;
wire [8:0] total_cycles = {1'b0, K_r} + MARGIN;

reg [1:0]  wr_row;

//====================================================================
// A/B 位址產生（純組合邏輯）
// A_index = m_tile * K + k   （對應 read_A_Matrix: mtile 外層、k 內層）
// B_index = n_tile * K + k   （對應 read_B_Matrix）
//====================================================================
wire               addr_valid_c = (state == S_FEED) && (cyc < {1'b0,K_r});
wire [7:0]         k_addr       = (cyc < {1'b0,K_r}) ? cyc[7:0] :
                                   (K_r == 8'd0 ? 8'd0 : K_r - 8'd1);

assign A_index = m_tile * K_r + k_addr;
assign B_index = n_tile * K_r + k_addr;

//====================================================================
// rd_valid：組合邏輯，跟 addr_valid_c 同拍（RD_LAT = 0）
//====================================================================
wire rd_valid = addr_valid_c;
reg  rdv1, rdv2, rdv3;   // 給 row1~3 / col1~3 用的延遲版本

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rdv1 <= 1'b0;
        rdv2 <= 1'b0;
        rdv3 <= 1'b0;
    end else begin
        rdv1 <= rd_valid;
        rdv2 <= rdv1;
        rdv3 <= rdv2;
    end
end

//====================================================================
// A/B 各 byte 的 skew shift register
// **byte 順序（依 PATTERN.v 反推）**：
//   [31:24] = offset0 (最小 index，不延遲)
//   [23:16] = offset1 (延遲1拍)
//   [15:8]  = offset2 (延遲2拍)
//   [7:0]   = offset3 (延遲3拍)
//====================================================================
reg [7:0] a1_d1, a2_d1, a2_d2, a3_d1, a3_d2, a3_d3;
reg [7:0] b1_d1, b2_d1, b2_d2, b3_d1, b3_d2, b3_d3;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        a1_d1<=0; a2_d1<=0; a2_d2<=0; a3_d1<=0; a3_d2<=0; a3_d3<=0;
        b1_d1<=0; b2_d1<=0; b2_d2<=0; b3_d1<=0; b3_d2<=0; b3_d3<=0;
    end else begin
        a1_d1 <= A_data_out[23:16];
        a2_d1 <= A_data_out[15:8];  a2_d2 <= a2_d1;
        a3_d1 <= A_data_out[7:0];   a3_d2 <= a3_d1; a3_d3 <= a3_d2;

        b1_d1 <= B_data_out[23:16];
        b2_d1 <= B_data_out[15:8];  b2_d2 <= b2_d1;
        b3_d1 <= B_data_out[7:0];   b3_d2 <= b3_d1; b3_d3 <= b3_d2;
    end
end

wire [7:0] row_in [0:3];
wire       row_vld[0:3];
wire [7:0] col_in [0:3];
wire       col_vld[0:3];

assign row_in[0] = A_data_out[31:24]; assign row_vld[0] = rd_valid;
assign row_in[1] = a1_d1;             assign row_vld[1] = rdv1;
assign row_in[2] = a2_d2;             assign row_vld[2] = rdv2;
assign row_in[3] = a3_d3;             assign row_vld[3] = rdv3;

assign col_in[0] = B_data_out[31:24]; assign col_vld[0] = rd_valid;
assign col_in[1] = b1_d1;             assign col_vld[1] = rdv1;
assign col_in[2] = b2_d2;             assign col_vld[2] = rdv2;
assign col_in[3] = b3_d3;             assign col_vld[3] = rdv3;

//====================================================================
// 4x4 PE array (output-stationary)
// A[i,k] 從第 i 列左側進入，延遲 i 拍進入 column0，之後每拍右移一格
// B[k,j] 從第 j 行上方進入，延遲 j 拍進入 row0，之後每拍下移一格
// => 在 PE(i,j) 對齊到 A[i,k]、B[k,j] 同時抵達 (時間 = k+i+j)
//====================================================================
reg [7:0]  a_reg [0:3][0:3];
reg        a_vld [0:3][0:3];
reg [7:0]  b_reg [0:3][0:3];
reg        b_vld [0:3][0:3];
reg [31:0] sum   [0:3][0:3];

integer ii, jj;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (ii = 0; ii < 4; ii = ii + 1) begin
            for (jj = 0; jj < 4; jj = jj + 1) begin
                a_reg[ii][jj] <= 8'd0;
                a_vld[ii][jj] <= 1'b0;
                b_reg[ii][jj] <= 8'd0;
                b_vld[ii][jj] <= 1'b0;
                sum[ii][jj]   <= 32'd0;
            end
        end
    end else if (state == S_INIT_TILE) begin
        for (ii = 0; ii < 4; ii = ii + 1) begin
            for (jj = 0; jj < 4; jj = jj + 1) begin
                a_reg[ii][jj] <= 8'd0;
                a_vld[ii][jj] <= 1'b0;
                b_reg[ii][jj] <= 8'd0;
                b_vld[ii][jj] <= 1'b0;
                sum[ii][jj]   <= 32'd0;
            end
        end
    end else if (state == S_FEED) begin
        for (ii = 0; ii < 4; ii = ii + 1) begin
            for (jj = 0; jj < 4; jj = jj + 1) begin
                if (a_vld[ii][jj] && b_vld[ii][jj]) begin
                    sum[ii][jj] <= sum[ii][jj] + a_reg[ii][jj]*b_reg[ii][jj];
                end

                if (jj == 0) begin
                    a_reg[ii][jj] <= row_in[ii];
                    a_vld[ii][jj] <= row_vld[ii];
                end else begin
                    a_reg[ii][jj] <= a_reg[ii][jj-1];
                    a_vld[ii][jj] <= a_vld[ii][jj-1];
                end

                if (ii == 0) begin
                    b_reg[ii][jj] <= col_in[jj];
                    b_vld[ii][jj] <= col_vld[jj];
                end else begin
                    b_reg[ii][jj] <= b_reg[ii-1][jj];
                    b_vld[ii][jj] <= b_vld[ii-1][jj];
                end
            end
        end
    end
end

//====================================================================
// 主控 FSM
//====================================================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state     <= S_IDLE;
        busy      <= 1'b0;
        C_wr_en   <= 1'b0;
        C_index   <= 16'd0;
        C_data_in <= 128'd0;
        K_r <= 0; M_r <= 0; N_r <= 0;
        m_tiles <= 0; n_tiles <= 0;
        m_tile  <= 0; n_tile  <= 0;
        cyc <= 0;
        wr_row <= 0;
    end else begin
        case (state)
            S_IDLE: begin
                C_wr_en <= 1'b0;
                if (in_valid) begin
                    K_r <= K; M_r <= M; N_r <= N;
                    m_tiles <= (M + 8'd3) >> 2;
                    n_tiles <= (N + 8'd3) >> 2;
                    m_tile <= 8'd0;
                    n_tile <= 8'd0;
                    busy   <= 1'b1;
                    state  <= S_INIT_TILE;
                end else begin
                    busy <= 1'b0;
                end
            end

            S_INIT_TILE: begin
                cyc     <= 9'd0;
                C_wr_en <= 1'b0;
                state   <= S_FEED;
            end

            S_FEED: begin
                cyc <= cyc + 9'd1;
                if (cyc + 9'd1 >= total_cycles) begin
                    state  <= S_WRITE;
                    wr_row <= 2'd0;
                end
            end

            S_WRITE: begin
                if ((m_tile * 8'd4 + wr_row) < M_r) begin
                    C_wr_en   <= 1'b1;
                    C_index   <= n_tile * M_r + (m_tile * 8'd4 + wr_row);
                    // C 封裝順序（依 PATTERN.v 反推）：
                    // [127:96]=offset0(最小n) ... [31:0]=offset3(最大n)
                    C_data_in <= { sum[wr_row][0], sum[wr_row][1],
                                   sum[wr_row][2], sum[wr_row][3] };
                end else begin
                    C_wr_en <= 1'b0;
                end

                if (wr_row == 2'd3) begin
                    state <= S_NEXT_TILE;
                end
                wr_row <= wr_row + 2'd1;
            end

            S_NEXT_TILE: begin
                C_wr_en <= 1'b0;
                if (m_tile == m_tiles - 8'd1) begin
                    m_tile <= 8'd0;
                    if (n_tile == n_tiles - 8'd1) begin
                        state <= S_DONE;
                    end else begin
                        n_tile <= n_tile + 8'd1;
                        state  <= S_INIT_TILE;
                    end
                end else begin
                    m_tile <= m_tile + 8'd1;
                    state  <= S_INIT_TILE;
                end
            end

            S_DONE: begin
                busy  <= 1'b0;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
        endcase
    end
end

endmodule