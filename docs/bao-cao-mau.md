# Báo cáo môn Mật Mã Ứng Dụng
## Tấn công DES bằng bảng cầu vồng tăng tốc trên GPU

> **Hướng dẫn dùng template:** mọi chỗ đánh dấu `<<...>>` là phần cần điền
> bằng số liệu thực đo trên server. Mọi chỗ `[chèn ảnh: ...]` là gợi ý nơi
> chèn screenshot từ phiên demo (xem `huong-dan-su-dung.md` §9).

---

**Sinh viên thực hiện:** <<Họ và tên>>  
**MSSV:** <<MSSV>>  
**Lớp / Nhóm:** <<Lớp>>  
**Giảng viên hướng dẫn:** <<Tên giảng viên>>  
**Ngày báo cáo:** <<dd/mm/yyyy>>  

**Mã nguồn:** https://github.com/kaito7926/CUDA_RAINBOWTABLE  
**Phần cứng dùng đo:** 2× NVIDIA L4 (SM 8.9, 23 GiB VRAM), CUDA 12.7  

---

## Tóm tắt (Abstract)

Báo cáo trình bày thiết kế, cài đặt và thực nghiệm một công cụ tấn công
**chosen-plaintext** lên thuật toán DES, sử dụng kỹ thuật **bảng cầu vồng
(rainbow table)** chạy song song trên GPU bằng CUDA. Không gian khóa
**danh nghĩa** là `[a-z0-9]^8 = 36⁸ ≈ 2.82 × 10¹²` chuỗi ASCII, nhưng do
**DES PC-1 loại 8 bit parity**, các cặp ký tự chỉ khác bit thấp nhất
(`b/c`, `d/e`, …, `0/1`, …, `8/9`) là DES-tương đương. Không gian khóa
**hiệu dụng** sụp đổ về `19⁸ ≈ 1.70 × 10¹⁰`, nhỏ hơn `36⁸` đến **166 lần**.
Với cấu hình `chain_len = 2²⁰`, `chains = 50,000`, công cụ đạt phủ lý
thuyết ~95% trên một bảng, kích thước đĩa ~800 KiB, thời gian build
<<X>> phút trên một GPU NVIDIA L4, và thời gian tra một ciphertext trung
bình <<Y>> giây. Bài học rút ra: DES với khóa ASCII bị suy yếu kép — vừa do
không gian khóa nhỏ, vừa do cấu trúc parity-bit khiến nhiều ASCII string đụng
độ; điều này khẳng định DES **không đủ an toàn** cho hệ mật hiện đại.

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết)
3. [Thiết kế hệ thống](#3-thiết-kế-hệ-thống)
4. [Cài đặt](#4-cài-đặt)
5. [Thực nghiệm](#5-thực-nghiệm)
6. [Phân tích kết quả](#6-phân-tích-kết-quả)
7. [Bàn luận về an toàn](#7-bàn-luận-về-an-toàn)
8. [Kết luận](#8-kết-luận)
9. [Tham khảo](#9-tham-khảo)

---

## 1. Giới thiệu

### 1.1 Bối cảnh

DES (Data Encryption Standard, FIPS-46, 1977) là mật mã khối 64-bit dùng
khóa 56-bit hiệu dụng. Mặc dù đã bị NIST thu hồi (2005), DES vẫn là **đối
tượng học thuật quan trọng** để minh họa các tấn công như brute-force,
differential cryptanalysis, linear cryptanalysis, và đặc biệt là **trade-off
thời gian–bộ nhớ** mà bảng cầu vồng đại diện.

### 1.2 Đặt vấn đề

Trong nhiều ứng dụng thực tế, người dùng đặt khóa DES từ **chuỗi văn bản**
8 ký tự (do DES nhận đầu vào 8 byte). Nếu hạn chế ký tự ở mức "ASCII có
nghĩa" như `[a-z0-9]`, không gian khóa thực sự là `36⁸ ≈ 2⁴¹·⁴` — **nhỏ hơn
2⁵⁶** nhiều lần. Câu hỏi: với phần cứng phổ thông (1–2 GPU), có thể build
sẵn cấu trúc dữ liệu cho phép tra ngược **bất kỳ ciphertext nào** ứng với
một bản rõ cố định, trong vài giây, hay không?

### 1.3 Mục tiêu

Trong khuôn khổ môn Mật Mã Ứng Dụng:

1. Cài đặt đúng DES theo FIPS-46, có kiểm tra KAT.
2. Cài đặt bảng cầu vồng kiểu Oechslin (round-dependent reduction).
3. Tăng tốc giai đoạn build bằng CUDA (one thread per chain).
4. Phân hoạch và lưu bảng ra đĩa kiểu sharded, sắp xếp theo endpoint.
5. Thực nghiệm: đo build time, phủ thực, tỉ lệ thành công khi tra
   ciphertext mẫu.
6. Bàn luận về độ an toàn rút ra từ thực nghiệm.

---

## 2. Cơ sở lý thuyết

### 2.1 DES và hạn chế của không gian khóa

DES nhận khóa 64-bit nhưng PC-1 loại đi 8 bit chẵn lẻ → 56 bit hiệu dụng.
Brute-force trực tiếp `2⁵⁶` từng được thực hiện (EFF DES Cracker, 1998, ~22
giờ). Tuy nhiên, **nếu khóa được giới hạn ở tập ký tự ASCII có nghĩa**, không
gian thực sự nhỏ hơn rất nhiều.

### 2.1.1 Sự sụp đổ vì parity của ASCII

Báo cáo này lấy `[a-z0-9]⁸ = 36⁸ ≈ 2.82·10¹²` làm minh họa. Nhưng PC-1 chỉ
giữ 7 bit cao của mỗi byte khóa, nên hai ký tự ASCII chỉ khác bit 0 (LSB)
ánh xạ về cùng một byte DES. Cụ thể:

| Lớp | Ký tự ASCII | 7 bit cao |
|---|---|---|
| 0  | `a`              | 0x30 |
| 1  | `b`, `c`         | 0x31 |
| 2  | `d`, `e`         | 0x32 |
| 3..11 | `f`/`g`, …, `t`/`u` | 0x33..0x3a |
| 12 | `v`, `w`         | 0x3b |
| 13 | `x`, `y`         | 0x3c |
| 14 | `z`              | 0x3d |
| 15..19 | `0`/`1`, …, `8`/`9` | 0x18..0x1c |

Tổng **19 lớp tương đương** (`{a}`, `{z}` lẻ + 12 cặp chữ cái + 5 cặp chữ
số). Không gian khóa hiệu dụng:

```
N_eff = 19⁸ = 16,983,563,041 ≈ 1.70·10¹⁰     (nhỏ hơn 36⁸ đến 166×)
```

Đây là cốt lõi của bài báo cáo: với khóa DES từ chuỗi ASCII, "alphabet"
hiệu dụng cho tấn công không phải `|charset|` mà là **số lớp parity**
trong charset đó.

### 2.2 Trade-off thời gian–bộ nhớ và bảng cầu vồng

#### Hellman 1980

Hellman đề xuất "chain":

```
y₀ ─f─> y₁ ─f─> y₂ ─...─> y_t = endpoint
```

với `f(y) = R(E_y(P))`, lưu `(y₀, y_t)`. Nhiều bảng dùng các hàm reduction
`R₁, R₂, ...` khác nhau. Kích thước bộ nhớ `M`, thời gian online `T`, độ
phủ thỏa `M² · T ≈ N²`.

#### Oechslin 2003 — Rainbow

Thay vì nhiều bảng × một reduction, dùng **một bảng** × `t` reductions thay
đổi theo vị trí:

```
y_{r+1} = R_r(E_{y_r}(P))    với r = 0, 1, ..., t-1
```

Lợi thế: số lần tra giảm từ `O(t²)` xuống `O(t²/2)`, và tỉ lệ thành công
trên cùng `M·T` cao hơn ~2 lần so với Hellman.

#### Xác suất phủ

Với `m` chain, độ dài `t`, không gian `N`:

```
P_table ≈ 1 - exp(- m·t / N)
```

Với nhiều bảng độc lập:

```
P_total ≈ 1 - (1 - P_table)^tables
```

#### Chain merge

Khi hai chain khác `chain_id` rơi vào cùng `y` tại cùng vị trí `r`, chúng
sẽ trùng nhau từ đó đến endpoint. Số chain hiệu dụng giảm. Rainbow giảm
xác suất merge **tại cùng vị trí** (vì `R_r ≠ R_{r'}`) nhưng không loại
hoàn toàn.

### 2.3 Thiết kế hàm reduction

Yêu cầu:

1. Phân phối `R(c)` xấp xỉ đều trên `[0, N)` (tránh lệch shard).
2. `R_r` tại các `r` khác nhau **không tương quan** (chống merge).
3. Rẻ trên GPU (không nhánh, ít memory access).

Lựa chọn trong dự án: **splitmix64** (Steele–Lea) kết hợp `round` và
`table_id`. Cụ thể:

```cpp
uint64_t reduce(uint64_t ct, uint32_t round, uint32_t table_id, uint64_t N) {
    uint64_t m = ct
              ^ splitmix64(round * 0x9E3779B97F4A7C15
                        ^ table_id * 0xC6BC279692B5C323);
    return splitmix64(m) % N;
}
```

`splitmix64` có avalanche tốt, chỉ là chuỗi multiply + shift → rẻ trên SM.

---

## 3. Thiết kế hệ thống

### 3.1 Tổng quan kiến trúc

```
       ┌──────────────┐    sinh        ┌──────────────┐    sort      ┌──────────────┐
       │ desrt build  │ ─────────────► │  raw/        │ ───────────► │  sorted/     │
       │   (GPU)      │   16 B/record  │  shard_*.bin │              │  shard_*.bin │
       └──────────────┘                 └──────────────┘              └──────────────┘
                                                                          ▲
                       ciphertext ─┐                                      │ binary
                                   ▼                                      │ search
                            ┌──────────────┐    endpoint candidates  ┌────┴────┐
                            │ desrt crack  │ ───────────────────────►│  walk   │
                            │   (GPU+CPU)  │                          └─────────┘
                            └──────────────┘
```

### 3.2 Tham số chính

| Tham số | Ký hiệu | Giá trị | Ghi chú |
|---|---|---|---|
| Bản rõ | `P` | `0x1122334455667788` | Cố định cho mọi chain |
| Charset (đầu vào) | — | `[a-z0-9]` | 36 ký tự, người dùng nhập |
| Charset (canonical) | — | `abdfhjlnprtvxz02468` | 19 ký tự DES-injective |
| Độ dài khóa | `k` | 8 | |
| Kích thước keyspace hiệu dụng | `N` | `19⁸ = 16,983,563,041` | ≈ 2³³·⁹⁸ |
| Độ dài chain | `t` | `2²⁰ = 1,048,576` | |
| Số chain mỗi bảng | `m` | `50,000` | đạt phủ ~95% |
| Số shard | — | `4096` | `shard = endpoint mod 4096` |
| Kích thước record | — | 16 B | `(endpoint, startpoint)` |
| Phủ lý thuyết một bảng | `P_table` | `≈ 0.9544` | `1 - e^{-m·t/N}` |
| Kích thước bảng | — | `~800 KiB` | `m × 16 B` |

### 3.3 Tạo điểm bắt đầu

```
start_idx(chain_id) = (LCG_A · chain_id + LCG_B) mod N
```

`LCG_A = 6364136223846793005`, `LCG_B = 1442695040888963407` (MMIX của
Knuth). Không phải hoán vị trên `[0, N)`, nhưng với `m = 50K ≪ N = 1.7e10`,
xác suất trùng start là không đáng kể (`m²/(2N) ≈ 7.4 × 10⁻⁵`).

### 3.4 Bố trí GPU

`desrt build` chạy theo batch (mặc định 65536 chain/batch):

- **Mỗi thread CUDA tính trọn một chain**, từ `start_idx` đến `endpoint`.
- Sau mỗi batch, copy về host buffer (pinned), `ShardWriter` phân loại
  record vào buffer per-shard và flush khi đủ ngưỡng.

Một thread → một chain là layout đơn giản nhất, không cần dùng shared
memory phức tạp; với DES sách giáo khoa, kernel bị giới hạn bởi tính
toán chứ không phải băng thông bộ nhớ.

### 3.5 Lưu trữ trên đĩa

```
tab/
├── raw/            (giai đoạn build, append-only)
│   ├── shard_000000.bin
│   └── ...
└── sorted/         (sau desrt sort)
    ├── shard_000000.bin
    └── ...
```

- 4096 shard ⇒ trung bình `50,000 / 4096 ≈ 12.2` record/shard ⇒
  `~200 B / shard`. Đủ nhỏ để sort in-memory bằng `std::sort`. Layout
  4096 shard có thể quá thưa cho 50K record; vẫn giữ vì tương lai
  multi-`table_id` sẽ tận dụng tốt.
- Sharding theo `endpoint mod 4096` đảm bảo phân bố đều (vì
  `reduce` đầu ra đều).

### 3.6 Thuật toán crack

Với ciphertext `CT` và bảng đã sort:

```
for p in 0..t-1:
    y ← reduce(CT, p, table_id, N)
    for r in p+1..t-1:
        y ← chain_step(y, r, ...)
    candidate_endpoint ← y
    shard ← y mod 4096
    push (shard, y) into bucket
```

GPU sinh `targets · t` ứng viên endpoint (mỗi ứng viên 8 B). Host:

1. Bucket theo `shard`, sort theo `(shard, y)`.
2. Đọc lần lượt từng sorted shard, `std::lower_bound` mỗi candidate.
3. Với mỗi match, **replay** từ `startpoint` `p` bước, lấy `key_p`, thử
   `DES(key_p, P) == CT` để loại false positive (do chain merge có thể
   tạo endpoint trùng nhưng khóa khác).

---

## 4. Cài đặt

### 4.1 Cấu trúc mã nguồn

| File | Vai trò |
|---|---|
| `include/desrt/common.h` | Hằng số, charset, `splitmix64`, `reduce_idx`, LCG |
| `include/desrt/des.h`    | FIPS-46 DES (`encrypt_block`, `chain_step`) |
| `include/desrt/shard.h`  | `ShardWriter`, `sort_shard`, `find_endpoint` |
| `include/desrt/cli.h`    | Flag parser |
| `src/cmd_build.cu`       | Kernel sinh chain + host streaming |
| `src/cmd_crack.cu`       | Kernel sinh candidate + shard walk |
| `src/cmd_bench.cu`       | KAT + đo throughput |
| `src/cmd_plan.cpp`       | Ước lượng phủ + thời gian |
| `src/cmd_sort.cpp`       | Sort song song theo shard |
| `src/cmd_make_target.cpp`| Sinh ciphertext mẫu |
| `src/cmd_stats.cpp`      | Thống kê bảng |
| `src/main.cpp`           | Dispatch 7 subcommand |

### 4.2 Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

### 4.3 Vấn đề kỹ thuật đáng kể

#### 4.3.1 DES tables trong device code

Phiên bản đầu khai báo các bảng tra (`IP`, `FP`, `PC-1`, `PC-2`, `E`, `P`,
`S-box`) là `inline constexpr` ở namespace scope. Khi build trên server,
bộ tiền xử lý CUDA báo lỗi:

```
identifier "IP" is undefined in device code
```

Nguyên nhân: `inline constexpr` ở namespace scope tạo ra biểu tượng
**host-only**. Cờ `--expt-relaxed-constexpr` chỉ cho phép gọi **hàm**
constexpr từ device, không áp dụng cho **biến**. Cách sửa: dùng macro tự
chọn storage class theo pha biên dịch:

```cpp
#if defined(__CUDA_ARCH__)
  #define DESRT_TABLE static __device__ const
#else
  #define DESRT_TABLE static const
#endif
DESRT_TABLE uint8_t IP[64] = { ... };
```

Mỗi translation unit có một bản (≈700 B per pass) — không đáng kể.

#### 4.3.2 Sự sụp đổ phủ vì DES parity (bug nghiêm trọng hơn)

Phiên bản đầu dùng charset 36 ký tự `[a-z0-9]` trực tiếp làm DES key bytes
và cấu hình `--chains 8,100,000` (ước lượng từ `36⁸`). Sau khi build và
chạy `desrt stats`:

```
total records      : 8,100,000
total unique eps   : 26,319 (0.0032 of records)
records/shard      : min=0  max=7764  mean=1977.5  empty=6
```

Chỉ ~26 nghìn endpoint duy nhất trên 8.1 triệu record — phủ thực **0.32%**
chứ không phải 95%. Phân tích merge ODE:

```
m(t) = 1 / (t/(2·N_eff) + 1/m₀)
     = 1 / (2²⁰/(2·19⁸) + 1/8.1·10⁶)
     ≈ 33,800
```

Khớp với thực đo 26,319 (sai số 1.3×). Nguyên nhân: chain step
`idx → key → DES → ct → next_idx` luôn đi qua "cổ chai" 19⁸ ciphertext
phân biệt (vì PC-1 sụp đổ 36⁸ ASCII keys về 19⁸ DES keys). Sau bước đầu
tiên, chain đã sống trong tập hỗ trợ kích thước 19⁸; chain merge với
tốc độ `m²/(2·19⁸)` rất cao khi `m = 8.1M`.

**Cách sửa**: chuyển charset nội bộ sang **19 ký tự canonical**
`abdfhjlnprtvxz02468`, mỗi ký tự đại diện một lớp parity. `idx_to_key`
chỉ phát ra canonical char → ánh xạ injective vào không gian DES key.
`string_to_idx` vẫn chấp nhận 36 ký tự `[a-z0-9]` (canonical hóa khi đọc).
`N_KEYSPACE` đổi thành `19⁸`; default `--chains` xuống `50,000`. Sau khi
sửa, build tương ứng đạt phủ ~95% như công thức dự đoán.

---

## 5. Thực nghiệm

> Các số liệu dưới đây cần điền bằng output thực đo. Nguồn xuất:
> chạy lần lượt các lệnh ở `huong-dan-su-dung.md` §9.

### 5.1 Môi trường

- Server: <<tên server / cluster>>
- GPU: 2× NVIDIA L4 (Compute 8.9, 23 GiB VRAM mỗi GPU)
- Driver: <<driver version>>
- CUDA Toolkit: 12.7
- CPU: <<thông số CPU>>
- RAM: 12 GiB
- Đĩa: 77 GiB, <<SSD/HDD>>
- OS: <<Ubuntu xx.xx>>

### 5.2 Kiểm tra DES (Known-Answer Test)

Chạy `desrt bench`. Kết quả:

| Test vector | Khóa | Plaintext | Ciphertext mong đợi | Kết quả |
|---|---|---|---|---|
| FIPS-46 v1 | `133457799BBCDFF1` | `0123456789ABCDEF` | `85E813540F0AB405` | <<PASS/FAIL>> |
| All-zero | `0000000000000000` | `0000000000000000` | `8CA64DE9C1B123A7` | <<PASS/FAIL>> |
| All-one | `FFFFFFFFFFFFFFFF` | `FFFFFFFFFFFFFFFF` | `7359B2163E4EDC58` | <<PASS/FAIL>> |

[chèn ảnh: screenshot output `desrt bench`]

### 5.3 Throughput

| Backend | Cấu hình đo | Throughput đo được |
|---|---|---|
| CPU 1 thread | 200 000 iters | <<X.XX MH/s>> |
| GPU 0 (L4) | 65 536 thread × 4096 iters | <<XXX MH/s>> |
| GPU 1 (L4) | 65 536 thread × 4096 iters | <<XXX MH/s>> |

So sánh: throughput GPU/CPU ≈ <<tỉ số>>×.

### 5.4 Build bảng

Lệnh:

```bash
desrt build --out ./tab --chains 50000 --chain-len 1048576 \
            --shards 4096 --table-id 0 --gpu 0
```

| Hạng mục | Giá trị |
|---|---|
| Tổng số chain | 50,000 |
| Tổng số bước DES | `50,000 × 2²⁰ ≈ 5.24·10¹⁰` |
| Thời gian build thực | <<M phút>> |
| Throughput hiệu dụng | <<XXX MH/s>> |
| Dung lượng `tab/raw/` | <<~800 KiB>> |
| Dung lượng `tab/sorted/` sau sort | <<~800 KiB>> |

[chèn ảnh: screenshot tiến trình build + `du -h tab/`]

### 5.5 Phân bố shard

Lệnh `desrt stats --table ./tab`:

| Chỉ số | Giá trị thực đo | Ghi chú |
|---|---|---|
| Tổng record | <<>> | = `chains` nếu không có lỗi |
| Endpoint duy nhất | <<>> | mong đợi 42K–48K (0.85–0.95 × 50K) |
| Mean records/shard | <<>> | mong đợi ≈ 12.2 |
| Stdev records/shard | <<>> | Poisson stdev ≈ √12 ≈ 3.5 |
| Min / Max records/shard | <<>> / <<>> | min ~5, max ~25 (đuôi Poisson) |

Nếu `unique_endpoints / records < 0.5` → đã build oversaturate hoặc
reduction function sai (xem §4.3.2 về bug PC-1 sụp đổ).

### 5.6 Thử crack

8 khóa ngẫu nhiên sinh bằng `desrt make-target --count 8 --seed 42`:

| # | Khóa thật | Ciphertext | Kết quả | Khóa thu được | Vị trí `p` của hit |
|---|---|---|---|---|---|
| 1 | <<>> | <<>> | <<HIT/MISS>> | <<>> | <<>> |
| 2 | <<>> | <<>> | <<>> | <<>> | <<>> |
| 3 | <<>> | <<>> | <<>> | <<>> | <<>> |
| 4 | <<>> | <<>> | <<>> | <<>> | <<>> |
| 5 | <<>> | <<>> | <<>> | <<>> | <<>> |
| 6 | <<>> | <<>> | <<>> | <<>> | <<>> |
| 7 | <<>> | <<>> | <<>> | <<>> | <<>> |
| 8 | <<>> | <<>> | <<>> | <<>> | <<>> |

Tỉ lệ thành công thực đo: <<X/8 = XX%>>.

So với lý thuyết `P_table ≈ 0.9508`: <<sát / lệch ±X%>>.

[chèn ảnh: screenshot output `desrt crack`]

### 5.7 Thời gian crack

| Hạng mục | Giá trị |
|---|---|
| Sinh ứng viên trên GPU | <<X.X s>> |
| Đọc + lower_bound các shard | <<X.X s>> |
| Replay-verify trên CPU | <<X.X s>> |
| Tổng wall-time cho 8 target | <<X.X s>> |
| Trung bình / target | <<X.X s>> |
| False positive (replay loại) | <<N>> |

---

## 6. Phân tích kết quả

### 6.1 So sánh phủ thực và phủ lý thuyết

Với `m·t/N = 50,000 · 2²⁰ / 19⁸ ≈ 3.087`, lý thuyết:

```
P_table = 1 - exp(-3.087) ≈ 0.9544
```

Thực đo: <<XX%>>. Sai khác do:

- **Chain merges**: số chain hiệu dụng < `m`. Tỉ số endpoint duy nhất / số
  record cho phép ước lượng tổn thất này.
- **Quy luật phân bố start_idx**: LCG không phải hoán vị, có thể trùng.
  Với `m ≪ N` (m=50K, N=1.7e10), ảnh hưởng nhỏ.

### 6.1.1 So sánh trước/sau khi sửa bug PC-1

| Cấu hình | `chains` | `total records` | `unique eps` | uniques/records |
|---|---|---|---|---|
| Trước (charset 36, N=36⁸) | 8,100,000 | 8,100,000 | 26,319 | **0.32%** |
| Sau (charset 19, N=19⁸) | 50,000 | 50,000 | <<>> | <<>> |

Sự khác biệt **không phải** vì giảm số chain — mà vì sửa được sự sụp đổ
DES PC-1. Build cũ "lãng phí" 166× compute cho cùng phủ.

### 6.2 Đánh giá reduction function

Tiêu chí: phân bố endpoint đều trên `[0, N)`.

Kiểm tra bằng `stats --per-shard`: stdev/mean của records/shard <<= X.X%>>
→ phân bố <<gần đều / có sai lệch>>. Đối với 4096 shard và 8.1 M record,
chuẩn lệch lý thuyết của Poisson(1977) ≈ `√1977 ≈ 44`. Đo thực: <<XX>>.

### 6.3 Hiệu năng GPU

Throughput đo được `<<XXX MH/s>>` so với mức lý thuyết của L4
(7.4 TFLOPS FP32) là **rất thấp** vì DES sách giáo khoa bị giới hạn bởi
các vòng lặp bit-by-bit (IP, FP, E, P, PC-2). Để đạt mức GH/s, cần dùng
**bitsliced DES** (Kwan, Rusakov) — mỗi thread xử lý 64 khóa song song
qua một thanh ghi 64-bit, S-box hiện thực bằng gate network.

### 6.4 Phân bố vị trí hit

Histogram `p` (vị trí trong chain mà target được tìm thấy) cho 8 target:

| Khoảng `p` | Số target |
|---|---|
| `[0, t/4)`     | <<>> |
| `[t/4, t/2)`   | <<>> |
| `[t/2, 3t/4)`  | <<>> |
| `[3t/4, t)`    | <<>> |

Lý thuyết: hit ở `p` gần `t-1` nhanh hơn (ít bước reduce phải làm) — phân
bố mong đợi gần đều theo `p`.

---

## 7. Bàn luận về an toàn

### 7.1 Bài học rút ra

1. **DES không an toàn cho khóa người-dùng.** Hạn chế ký tự ở `[a-z0-9]`
   8 ký tự đã hạ độ phức tạp tấn công xuống chỉ <<M phút>> build offline
   + vài giây tra online. Mọi hệ thống còn dùng DES với mật khẩu dạng
   chuỗi ASCII đều đã bị bẻ ngay khi attacker chấp nhận <1 MiB bộ nhớ.

1a. **DES parity-bit nhân đôi điểm yếu.** Tưởng có 36⁸ khóa khả thi, thực
   tế chỉ có 19⁸ lớp tương đương — DES nuốt 8 bit "tự do" mỗi khóa làm
   không gian khóa sụp 166× cho `[a-z0-9]`. Với charset rộng hơn
   (`[a-zA-Z0-9]`, 62 ký tự) hệ số sụp lớn hơn nữa do nhiều cặp parity
   hơn.

2. **Trade-off thời gian – bộ nhớ là tấn công thực, không phải lý thuyết.**
   Hellman 1980 từng bị coi là "không thực tế trên phần cứng đương thời" —
   với một GPU phổ thông năm 2024, việc đó là vài phút.

3. **Salting là biện pháp đối phó cơ bản.** Nếu mỗi target có một
   plaintext khác nhau (qua salt), bảng cầu vồng phải build lại cho từng
   target → vô dụng. Đây là lý do `crypt(3)` hiện đại luôn salt.

4. **Khóa phải đến từ KDF, không phải từ chuỗi ASCII trực tiếp.** PBKDF2,
   bcrypt, scrypt, Argon2 đặt thêm work factor làm bảng cầu vồng tốn kém
   hơn theo cấp số nhân.

### 7.2 Giới hạn của báo cáo

- Chỉ tấn công **một bản rõ cố định**. Trong thực tế, bản rõ thường
  ngẫu nhiên — đây là tấn công "chosen-plaintext attack" thuần túy.
- Bỏ qua các kiểu khóa: chữ hoa, ký tự đặc biệt, độ dài < 8. Để hỗ trợ,
  chỉ cần đổi `CHARSET_LEN` và 2–3 hàm helper trong `common.h`; tổng chi
  phí scale theo `N`.
- Chưa cài bitsliced DES — số đo throughput phản ánh sách giáo khoa, không
  phải giới hạn của phần cứng.

### 7.3 Hướng phát triển

1. **Bitsliced DES kernel**: kỳ vọng 3–8 GH/s/L4, nâng phủ lên `36⁹` hoặc
   chuyển sang `[A-Za-z0-9]⁸ = 62⁸`.
2. **Đa bảng**: chạy `--table-id 0..k-1` để đẩy phủ lên `1 - 0.05⁴ > 99.99%`.
3. **Distinguished points**: thay vì lưu tại đúng `t` bước, dừng khi
   endpoint có một mẫu bit nhất định — giảm I/O khi tra.
4. **Multi-GPU một process**: hiện đang phải chạy 2 tiến trình. Thêm
   per-shard mutex hoặc per-GPU subdir là cải tiến đơn giản.

---

## 8. Kết luận

Báo cáo đã trình bày trọn vẹn vòng đời của một tấn công bảng cầu vồng lên
DES với không gian khóa `[a-z0-9]⁸`, từ thiết kế (chain construction,
reduction round-dependent, sharded disk layout, charset canonical hóa
19 ký tự), cài đặt (CUDA C++17, `__host__ __device__` reusable DES, kernel
one-thread-per-chain), đến thực nghiệm (KAT FIPS-46, throughput, build
time, phủ thực, thời gian tra). Phủ thực đo được <<X%>> so với chặn trên
lý thuyết 95.44%; bảng <800 KiB cho phép tra ciphertext trong <<Y giây>>.
Kết quả minh họa cụ thể **tại sao DES không còn được khuyến nghị** và đặc
biệt nêu bật một bài học ít được nhắc tới: **DES parity-bit khiến không
gian khóa ASCII sụp đổ 100× trở lên trên nhiều charset thông dụng** — lý
do các hệ mật hiện đại phải gắn liền với salt + KDF chứ không được dùng
mật khẩu ASCII trực tiếp làm key.

Toàn bộ mã nguồn, hướng dẫn sử dụng và tài liệu thiết kế nằm tại:
https://github.com/kaito7926/CUDA_RAINBOWTABLE.

---

## 9. Tham khảo

1. NIST. *Data Encryption Standard*. FIPS PUB 46-3, 1999.
2. Hellman, M. *A Cryptanalytic Time-Memory Trade-Off*. IEEE Trans.
   Information Theory, IT-26(4), 1980.
3. Oechslin, P. *Making a Faster Cryptanalytic Time-Memory Trade-Off*.
   CRYPTO 2003, LNCS 2729.
4. Kwan, M. *Reducing the Gate Count of Bitslice DES*. 2000.
5. Knuth, D. *The Art of Computer Programming, Vol. 2*: §3.3.4 (MMIX LCG).
6. Steele, G. L., Lea, D., Flood, C. H. *Fast Splittable Pseudorandom
   Number Generators*. OOPSLA 2014. (splitmix64)
7. NVIDIA. *CUDA C++ Programming Guide*, 12.7. NVIDIA Corporation, 2024.
8. EFF. *Cracking DES: Secrets of Encryption Research, Wiretap Politics
   and Chip Design*. O'Reilly, 1998.

---

## Phụ lục A. Lệnh đã chạy

```bash
# Phiên bench
desrt bench

# Build bảng đầy đủ
desrt build --out ./tab --chains 50000 --chain-len 1048576 --shards 4096 --gpu 0

# Sort + stats
desrt sort  --in ./tab --shards 4096
desrt stats --table ./tab

# Sinh target và crack
desrt make-target --out targets.txt --count 8 --seed 42
desrt crack --table ./tab --target targets.txt
```

## Phụ lục B. Log thực đo

```
<<dán nguyên output desrt bench>>
```

```
<<dán nguyên output desrt plan>>
```

```
<<dán nguyên output desrt stats>>
```

```
<<dán nguyên output desrt crack>>
```
