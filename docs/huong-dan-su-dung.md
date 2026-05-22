# desrt — Hướng dẫn sử dụng (môn Mật Mã Ứng Dụng)

Tài liệu này hướng dẫn từng bước cách dùng `desrt` để **xây dựng và khai thác
một bảng cầu vồng (rainbow table) tấn công DES** trong một bài demo môn học.

> Đối tượng: sinh viên Mật Mã Ứng Dụng đã biết DES ở mức sách giáo khoa
> (FIPS-46), chưa cần biết CUDA. Mọi câu lệnh CUDA đều chạy trên **server**;
> máy lập trình chỉ dùng để chỉnh sửa code và đẩy lên git.

---

## 0. Hiểu nhanh repo đang làm gì

Bài toán: cho trước **bản rõ cố định** `P = 0x1122334455667788`, kẻ tấn công
muốn phục hồi khóa DES `K` (8 ký tự ASCII trong `[a-z0-9]`) từ ciphertext
`C = DES(K, P)`.

Không gian khóa — chú ý cú twist của DES:

```
alphabet người dùng = "abcdefghijklmnopqrstuvwxyz0123456789"   (36 ký tự)
key_len             = 8
N (naive)           = 36^8 = 2,821,109,907,456  ≈ 2^41.4
```

Nhưng **DES PC-1 bỏ bit parity** (bit 0 của mỗi byte khóa) → các cặp ký tự
chỉ khác bit thấp nhất là tương đương dưới DES (`b/c`, `d/e`, `f/g`, …,
`x/y`, `0/1`, `2/3`, …, `8/9`). Trong `[a-z0-9]` chỉ có **19 lớp tương
đương** (14 chữ cái + 5 chữ số):

```
charset canonical   = "abdfhjlnprtvxz02468"   (|charset| = 19)
N hiệu dụng         = 19^8 = 16,983,563,041  ≈ 2^33.98
```

`desrt` lưu chain theo `N = 19⁸`. Người dùng vẫn nhập bất kỳ ký tự nào trong
36 chữ `[a-z0-9]` ở `make-target --key …`; `string_to_idx` tự canonical hóa
về 19 ký tự. Khóa được khôi phục sẽ ở dạng canonical (DES-tương đương).

Cách làm: xây **rainbow table** offline (tốn thời gian + đĩa) rồi tra
ciphertext online (nhanh). Mỗi "chain" trong bảng:

```
start_idx ─DES─> ct₀ ─reduce─> idx₁ ─DES─> ct₁ ─reduce─> idx₂ ─...─> endpoint
```

Bảng chỉ lưu `(endpoint, start_idx)` cho mỗi chain → tradeoff giữa bộ nhớ và
thời gian (time–memory trade-off, Hellman 1980; rainbow Oechslin 2003).

7 lệnh con của `desrt`:

| Lệnh | Mục đích |
|---|---|
| `desrt plan`        | In ra dự kiến phủ, dung lượng, thời gian build. Không ghi đĩa. |
| `desrt bench`       | Kiểm tra DES (KAT FIPS-46) + đo throughput. |
| `desrt build`       | Sinh bảng trên GPU, ghi shard thô ra đĩa. |
| `desrt sort`        | Sắp xếp các shard theo endpoint. |
| `desrt stats`       | Thống kê số bản ghi, endpoint duy nhất. |
| `desrt make-target` | Sinh ciphertext mẫu để test crack. |
| `desrt crack`       | Tra một hoặc nhiều ciphertext về khóa. |

Luồng demo điển hình: **plan → bench → build → sort → stats → make-target →
crack**.

---

## 1. Chuẩn bị server

Cấu hình tham chiếu (đề bài):

- 2× NVIDIA L4 (SM 8.9, 23 GiB VRAM mỗi GPU)
- CUDA 12.7, driver tương ứng
- RAM 12 GiB, đĩa 77 GiB
- Linux x86_64

Cài đặt:

```bash
nvidia-smi               # xác nhận thấy cả 2 GPU
nvcc --version           # phải là 12.x
cmake --version          # >= 3.18
```

Clone repo (tên repo trên GitHub là `CUDA_RAINBOWTABLE`):

```bash
git clone https://github.com/kaito7926/CUDA_RAINBOWTABLE.git
cd CUDA_RAINBOWTABLE
```

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

Sau khi build thành công, binary nằm ở `./build/desrt`. Tiện nhất là thêm
alias:

```bash
alias desrt=./build/desrt
desrt --help
```

> **Lưu ý:** Máy lập trình (Windows) **không cần cài CUDA**. Mọi lệnh `desrt`
> bên dưới đều chạy trên server.

---

## 2. `desrt plan` — Ước lượng trước khi tốn GPU

```bash
desrt plan
```

Output mẫu (giá trị thực tế lấy tại server):

```
desrt plan
  charset (canonical): abdfhjlnprtvxz02468 (|charset|=19)
  accepts at input   : [a-z0-9] (auto-canonicalised in string_to_idx)
  key length         : 8
  N (keyspace)       : 16983563041 (19^8, DES-injective)
  chain_len          : 1048576
  chains             : 50000
  tables             : 1
  shards             : 4096
  rate               : 10.00 GH/s (giả định)

  L * C / N           ≈ 3.087
  p_one_table         ≈ 0.9544     (≈95% upper bound)
  raw size per table  ≈ 0.8 MiB
  total DES ops       ≈ 5.24e+10
  est. build time     ≈ 5.24 s     (ở 10 GH/s)
```

Ý nghĩa các con số:

- `L * C / N ≈ 3.087` — trung bình mỗi vị trí của không gian khóa được
  "thăm" ≈ 3 lần. Phủ cao, nhưng vẫn có chain merge.
- `p_one_table ≈ 0.9544` — chặn trên của xác suất bẻ một khóa ngẫu nhiên
  bằng **một** bảng. Thực tế thấp hơn (~85–95% của con số này).
- `est. build time` — chỉ là ước lượng theo `--rate 10 GH/s`. Số thực
  trên L4 ~150 MH/s sẽ là vài phút.

Tham số quan trọng để thử:

```bash
desrt plan --chain-len 524288 --chains 100000        # chain ngắn hơn, nhiều chain hơn
desrt plan --tables 2                                # 2 bảng, mỗi bảng table_id khác nhau
```

---

## 3. `desrt bench` — Kiểm tra DES + đo throughput

**Luôn chạy lệnh này đầu tiên** sau khi build. Nếu DES sai, không có gì
hoạt động đúng.

```bash
desrt bench
```

Output mẫu:

```
KAT FIPS-46 vector 1: PASS  (key=133457799BBCDFF1 pt=0123456789ABCDEF -> 85E813540F0AB405)
KAT all-zero:        PASS
KAT all-one:         PASS

Host (1 thread) :  3.42 MH/s  (200000 iters in 58.5 ms)
GPU 0           :  152.4 MH/s (65536 threads * 4096 iters in 1.76 s)
```

- Nếu bất kỳ KAT nào FAIL → exit code 3, **dừng lại**. Đừng build bảng. Xem
  lại file `include/desrt/des.h`.
- Throughput GPU dao động 50–300 MH/s/L4 cho DES sách giáo khoa. Thấp hơn
  10 GH/s spec ban đầu (spec đó cho không gian 62⁸ lớn hơn ~3000 lần),
  nhưng với `N_eff = 19⁸` chỉ cần ~5·10¹⁰ DES op để build 50K chain × 2²⁰
  steps → vài phút trên một L4.

Các flag hữu ích:

```bash
desrt bench --gpu 1                # dùng GPU thứ hai
desrt bench --gpu-iters 16384      # đo dài hơn để có số ổn định
desrt bench --skip-host            # bỏ qua phần CPU nếu không cần
```

---

## 4. `desrt build` — Sinh bảng (giai đoạn tốn GPU nhất)

### 4.1 Smoke test trước

Trước khi chạy bảng đầy đủ, sinh một bảng nhỏ để chắc rằng pipeline đúng:

```bash
mkdir -p tab_smoke
desrt build --out ./tab_smoke --chains 100000 --chain-len 65536 --shards 256
```

Output mẫu:

```
desrt build: out=./tab_smoke chains=100000 chain_len=65536 shards=256
            batch_size=65536 table_id=0 gpu=0
launch 1/2  chains 0..65535       1.71 s
launch 2/2  chains 65536..99999   0.92 s
flushed 100000 records into 256 raw shards
```

→ tạo cây thư mục:

```
tab_smoke/
└── raw/
    ├── shard_000000.bin
    ├── shard_000001.bin
    └── ... (256 file, mỗi file khoảng 6 KiB)
```

### 4.2 Bảng đầy đủ cho demo

Với keyspace hiệu dụng `19^8`, bảng đầy đủ ~0.8 MiB, build vài phút trên L4:

```bash
mkdir -p tab
desrt build --out ./tab \
    --chains 50000 \
    --chain-len 1048576 \
    --shards 4096 \
    --table-id 0 \
    --gpu 0
```

Thời gian thực tế: tùy throughput L4 (~150 MH/s với DES sách giáo khoa),
khoảng 5–10 phút.

> **Cẩn thận:** đừng đặt `--chains 8100000` như draft trước đó. `N_eff = 19^8`
> nhỏ hơn 36^8 đến 166×, nên 8.1M chain sẽ oversaturate, chain merge hàng
> loạt, và `desrt stats` sẽ chỉ ra ~26K endpoint duy nhất trên 8.1M record
> (xem §11 lỗi thường gặp).

### 4.3 Chia việc cho 2 GPU (tùy chọn)

Mỗi GPU làm một nửa số chain, ghi vào cùng thư mục:

```bash
desrt build --out ./tab --start-chain-id 0     --chains 25000 --gpu 0 &
desrt build --out ./tab --start-chain-id 25000 --chains 25000 --gpu 1 &
wait
```

Hai tiến trình ghi append vào `./tab/raw/shard_*.bin`. Sau khi cả hai xong,
chạy `desrt sort` **một lần** là gộp xong.

### 4.4 Resume khi gián đoạn

Nếu cần dừng và chạy tiếp:

1. Đếm bao nhiêu chain đã ghi (qua `desrt stats --table ./tab`).
2. Chạy lại với `--start-chain-id <số_đã_xong>`.

Vì `chain_id` quyết định hoàn toàn nội dung chain, hai lần chạy không
overlap sẽ ra đúng bảng.

---

## 5. `desrt sort` — Sắp xếp theo endpoint

`desrt crack` cần các shard đã sắp xếp để binary-search.

```bash
desrt sort --in ./tab --shards 4096
```

Output mẫu:

```
desrt sort: in=./tab shards=4096 jobs=8
  sorted shard 0000..0511   (8.45 s)
  sorted shard 0512..1023   (8.31 s)
  ...
done. wrote 4096 sorted shards, deleted 4096 raw shards
```

Sau bước này:

```
tab/
└── sorted/
    ├── shard_000000.bin
    └── ...
```

Nếu muốn giữ shard thô (debug):

```bash
desrt sort --in ./tab --keep-raw
```

---

## 6. `desrt stats` — Kiểm tra bảng đã build

```bash
desrt stats --table ./tab
```

Output mẫu (build "khỏe"):

```
desrt stats: table=./tab shards=4096 jobs=24 (read in 0.05s)
  total records      : 50000
  total unique eps   : ~42000–48000  (0.85–0.95 of records)
  raw bytes (16B/rec): 0.0008 GiB
  records/shard      : min ~5  max ~20  mean=12.2  empty=~few
```

Đọc các con số:

- **unique endpoints / records** càng gần 100% càng tốt. Mức ~85–95% là bình
  thường ở `L·C/N ≈ 3`; phần thiếu là chain merges.
- **records/shard** phân bố theo Poisson(mean=12.2), kỳ vọng stdev ≈ √12 ≈
  3.5 → min/max nằm trong vài lần stdev. Nếu max >> mean nhiều thì phân
  phối endpoint không đều → kiểm tra reduction function.

> **Triệu chứng "build bị bệnh"**: nếu `total unique eps` rất nhỏ so với
> `total records` (ví dụ 0.003 of records), chain đang merge thảm khốc. Xem
> §11 "Lỗi thường gặp" — gần như chắc chắn do build oversaturate (chains
> nhiều hơn `3 · N_eff / chain_len`).

Thêm `--per-shard` để in từng dòng cho mỗi shard (debug phân bố):

```bash
desrt stats --table ./tab --per-shard | head
```

---

## 7. `desrt make-target` — Sinh ciphertext mẫu để demo

Sinh 8 khóa ngẫu nhiên + ciphertext tương ứng:

```bash
desrt make-target --out targets.txt --count 8
```

`targets.txt`:

```
1234567890  abcd1234  3F8A9C20D1E47B5C
9876543210  ze9xq0mz  91F2E58A0C4D6677
...
```

Mỗi dòng: `<idx>\t<key>\t<ciphertext_hex>`.

Sinh đúng một target cụ thể (để dự đoán được vị trí trúng):

```bash
desrt make-target --out one.txt --key "hello123"
desrt make-target --out one.txt --key-index 0
```

Khóa với ký tự ngoài `[a-z0-9]` sẽ bị từ chối:

```bash
desrt make-target --key "Hello123"
# make-target: --key must be exactly 8 characters from [a-z0-9]
```

Khóa có ký tự "không canonical" (trong `[a-z0-9]` nhưng không thuộc 19 chữ
canonical) thì được **chấp nhận và tự canonical hóa**. Ví dụ:

```bash
desrt make-target --out one.txt --key "abcdefgh"
# wrote one.txt — key column = "abddffhh" (canonical DES-equivalent của abcdefgh)
```

Đây là điểm mấu chốt của bài học DES parity: `'c'` và `'b'` đi qua PC-1 ra
cùng một byte, nên crack chỉ có thể trả về dạng canonical.

---

## 8. `desrt crack` — Tra ciphertext về khóa

```bash
desrt crack --table ./tab --target targets.txt
```

Output mẫu (theo đúng định dạng `cmd_crack.cu`):

```
desrt crack: table=./tab targets=8 chain_len=1048576 shards=4096
GPU 0: computed 8388608 endpoint candidates in 4.7 s
host: walked 4096 shards, 31 false positives, lookup phase 2.1 s

Results (7/8 solved):
  [HIT ] abcd1234  ct=3F8A9C20D1E47B5C  key=abcd1234  idx=1234567890
  [HIT ] ze9xq0mz  ct=91F2E58A0C4D6677  key=ze9xq0mz  idx=9876543210
  ...
  [MISS] zzzzzzzz  ct=...
total wall time: 6.9 s
```

Diễn giải:

- **HIT** — replay-verify thành công, đúng khóa (có thể là khóa DES-tương
  đương vì DES bỏ parity).
- **MISS** — không tìm được. Bình thường ~5% nếu coverage ~95%. Nếu nhiều
  hơn dự đoán, kiểm tra `--chain-len` / `--table-id` / `--plaintext` có
  khớp lúc build không.
- **false positives** — số endpoint trùng nhưng replay sai. Đã được lọc
  trên CPU bằng `DES(key_ứng_viên, plaintext) == CT`.

Crack một ciphertext lẻ không qua file:

```bash
desrt crack --table ./tab --ct 3F8A9C20D1E47B5C
```

Exit code: 0 nếu tất cả target tìm được, 4 nếu có ít nhất một MISS.

---

## 9. Demo end-to-end (chép đúng để chạy)

Toàn bộ demo, ~10–20 phút tùy GPU:

```bash
# 0. Build binary
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
alias desrt=./build/desrt

# 1. Sanity-check: DES KAT + throughput
desrt bench

# 2. Xem dự kiến
desrt plan

# 3. Smoke test pipeline với bảng tí hon (~1 giây)
desrt build --out ./tab_smoke --chains 2000 --chain-len 4096 --shards 64
desrt sort  --in  ./tab_smoke --shards 64
desrt stats --table ./tab_smoke --shards 64

# 4. Bảng demo đầy đủ (~5–10 phút)
desrt build --out ./tab --chains 50000 --chain-len 1048576 --shards 4096
desrt sort  --in  ./tab --shards 4096
desrt stats --table ./tab

# 5. Sinh và crack target
desrt make-target --out targets.txt --count 16 --seed 42
desrt crack --table ./tab --target targets.txt
```

Chụp màn hình hoặc lưu lại output của bước 1, 2, 3, 4, 5 — đó chính là bằng
chứng cho báo cáo môn học (xem `bao-cao-mau.md`).

---

## 10. Mở rộng cho slide / câu hỏi giám khảo

### "Vì sao charset là 19 ký tự chứ không phải 36?"

DES PC-1 bỏ bit 0 (LSB) của mỗi byte khóa. Trong ASCII, các cặp chỉ khác bit
0 (`b`/`c`, `d`/`e`, `0`/`1`, …) bị PC-1 ánh xạ về cùng một byte khóa.
Trong `[a-z0-9]` có 12 cặp chữ cái + 5 cặp chữ số + `a`, `z` đứng một mình
→ chỉ 19 lớp tương đương. Nếu chạy chain trên 36⁸ idx, mọi bước đều bị nén
qua "cổ chai" 19⁸ ciphertext khác nhau, nên chain merge với tốc độ
`m² / (2 · 19⁸)` rất nhanh — sau 2²⁰ bước hầu hết chain đã collapse.

Đây là bài học cốt lõi: **không gian khóa thực của DES với ASCII là 19⁸ chứ
không phải 36⁸**. Khóa khôi phục được là một đại diện canonical (DES-tương
đương), không nhất thiết là chuỗi ASCII gốc người dùng nhập.

### "Vì sao chain merges làm giảm phủ?"

Hai chain khác `chain_id` có thể đi qua cùng một `idx` tại cùng vị trí `r`.
Từ vị trí đó trở đi chúng trùng nhau hoàn toàn. Số chain "hiệu dụng" giảm.
Reduction phụ thuộc `round` giúp **giảm xác suất collision tại cùng vị trí**,
không loại trừ hoàn toàn.

### "Vì sao bảng lưu `(endpoint, start)` thay vì cả chain?"

Lưu cả chain = `chain_len × 8 B` mỗi chain = lãng phí. Mỗi chain có thể
**dựng lại** từ `start_idx` chỉ bằng DES + reduce, nên chỉ cần lưu hai đầu.
Khi crack có endpoint trùng, replay từ `start` đến vị trí cần tìm.

### "Vì sao reduce phải XOR với `round`?"

Không có `round` → mỗi `ct` luôn đi về cùng `idx` → mọi chain crashes thành
một chain duy nhất. Đây là rainbow vs Hellman classic: Hellman dùng nhiều
bảng với `f_i` khác nhau; rainbow nhúng `round` vào `f` để dùng một bảng.

### "DES dùng parity bit không?"

Không. PC-1 loại bit thứ 8, 16, ..., 64. Nên hai khóa `K` và `K ⊕ parity_mask`
ra cùng ciphertext. `desrt crack` có thể trả về **khóa DES-tương đương** —
encrypt thử nghiệm vẫn ra đúng ciphertext, nhưng chuỗi ASCII có thể khác
khóa người dùng đặt.

### "Vì sao chọn `chain_len = 2^20`?"

Tradeoff:

- Chain dài → ít chain hơn cho cùng phủ → bảng nhỏ hơn (ít record).
- Chain dài → crack chậm hơn (phải thử nhiều vị trí `p`).
- Chain dài → nhiều merges (xác suất 2 chain gặp nhau tỉ lệ với `L²`).

`2^20` là điểm thường được dùng cho keyspace cỡ `2^40`.

---

## 11. Lỗi thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `identifier ... is undefined in device code` cho `IP`, `PC1`, ... | Build cũ trước commit `02ae9de`. Pull mới nhất rồi build lại. |
| KAT FIPS-46 FAIL | Có chỉnh tay `des.h` → kiểm tra lại bit numbering (bit 1 = MSB). |
| `desrt crack` ra 0 hit dù bảng đã build | `--chain-len` / `--table-id` / `--plaintext` lúc crack không khớp lúc build. |
| **`desrt stats` báo `unique eps ≈ 0.003 of records`** (kiểu `26319 / 8100000`) | **Oversaturate**: chạy `--chains 8100000` trên keyspace `19⁸ ≈ 1.7e10`. `L·C / N ≈ 500` chứ không phải 3 → chain merge thảm khốc. Sửa: rebuild với `--chains 50000` (mặc định mới). |
| `desrt stats` báo `unique endpoints` thấp bất thường nhưng `chains` ở mức hợp lý | Reduction sai (không nhân `round` vào). Kiểm tra `reduce_idx` trong `common.h`. |
| OOM trên L4 khi `desrt crack` | Giảm `--block-size`, hoặc chia `targets.txt` thành nhiều phần. |
| `make-target: --key must be exactly 8 characters from [a-z0-9]` | Có chữ hoa hoặc ký hiệu — charset hiện tại không nhận. |
| `make-target` trả về key khác chuỗi vừa nhập (vd `abcdefgh` → `abddffhh`) | Đúng — đã được canonical hóa về 19 ký tự DES-injective. Hai chuỗi là DES-tương đương. |

---

## 12. Đọc thêm trong repo

- `docs/design.md` — toán học sau các con số, giải thích layout GPU + disk.
- `docs/commands.md` — bảng tham khảo tham số chi tiết.
- `include/desrt/des.h` — DES sách giáo khoa, tham chiếu thẳng FIPS-46.
- `include/desrt/common.h` — toàn bộ hằng số, charset, reduction function.
- `src/cmd_*.cpp` / `src/cmd_*.cu` — mỗi lệnh con là một file.
