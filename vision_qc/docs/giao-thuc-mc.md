# Giao thức MC (SLMP) — tham chiếu đủ để tự viết lại

Tài liệu này ghi lại đúng những gì `vision_qc/mc.py` cài đặt, để người đọc
không phải mở tài liệu Mitsubishi SH-080008 (hơn 700 trang) ra tra.

Đây là giao thức PLC Mitsubishi dùng để nói chuyện qua Ethernet. FX5U, Q
series, L series, iQ-R, iQ-F đều nói được. Không cần thư viện: khung tin là
một chuỗi byte cố định gửi qua TCP.

---

## 1. Khung 3E (binary)

Có bốn loại khung: 1E, 2E, 3E, 4E. **3E** là loại thông dụng nhất và là loại
duy nhất `mc.py` cài đặt. 4E thêm 2 byte số thứ tự ở đầu; 1E/2E là bản cũ.

### Yêu cầu

| Offset | Nội dung | Byte | Ví dụ | Ghi chú |
| ---: | --- | ---: | --- | --- |
| 0 | Subheader | 2 | `50 00` | 3E request |
| 2 | Network No. | 1 | `00` | |
| 3 | PC No. | 1 | `FF` | `FF` = PLC nội bộ |
| 4 | Module I/O No. | 2 | `FF 03` | `0x03FF`, lưu little-endian |
| 6 | Module Station No. | 1 | `00` | |
| 7 | Request data length | 2 | `0C 00` | từ **monitoring timer** tới hết |
| 9 | Monitoring timer | 2 | `10 00` | `0x0010` × 250 ms = 4 giây |
| 11 | Command | 2 | `01 04` | `0x0401` = đọc hàng loạt |
| 13 | Subcommand | 2 | `00 00` | `0x0000` theo từ, `0x0001` theo bit |
| 15 | Dữ liệu | — | | tuỳ lệnh |

### Trả lời

| Offset | Nội dung | Byte | Ví dụ |
| ---: | --- | ---: | --- |
| 0 | Subheader | 2 | `D0 00` |
| 2 | Network No. | 1 | `00` |
| 3 | PC No. | 1 | `FF` |
| 4 | Module I/O No. | 2 | `FF 03` |
| 6 | Module Station No. | 1 | `00` |
| 7 | Response data length | 2 | `04 00` |
| 9 | **End code** | 2 | `00 00` = thành công |
| 11 | Dữ liệu | — | |

### Chỗ dễ sai nhất: độ dài

`request data length` đếm từ **monitoring timer**, tức là **offset 9**, chứ
không phải từ đầu khung. Với khung yêu cầu, 15 byte đầu đã bao gồm 6 byte đó
(timer 2 + command 2 + subcommand 2), nên:

```text
độ dài = 6 + số byte dữ liệu
```

Lỗi này đã thật sự xảy ra khi viết `SimMcPlc`: bên nhận đọc 15 byte rồi đọc
thêm `declared` byte nữa, tức là đọc thừa đúng 6 byte, và treo cho tới khi hết
thời gian chờ. Triệu chứng là "PLC đóng kết nối giữa chừng" — một thông báo
chẳng liên quan gì tới nguyên nhân thật.

Với khung trả lời, `response data length` = 2 (end code) + số byte dữ liệu.

---

## 2. Lệnh

| Command | Sub | Việc |
| --- | --- | --- |
| `0x0401` | `0x0000` | đọc hàng loạt, theo từ |
| `0x0401` | `0x0001` | đọc hàng loạt, theo bit |
| `0x1401` | `0x0000` | ghi hàng loạt, theo từ |
| `0x1401` | `0x0001` | ghi hàng loạt, theo bit |

### Dữ liệu của lệnh đọc/ghi hàng loạt

```text
mã thiết bị (1 byte)
đầu thiết bị (3 byte, little-endian)
số điểm     (2 byte, little-endian)
```

Khi **ghi**, theo sau là các giá trị:

- theo từ: 2 byte mỗi từ, little-endian, **số có dấu bù hai**
- theo bit: `ceil(số_điểm / 2)` byte, **2 bit gói vào 1 byte**, bit thứ nhất ở
  nửa CAO

Ví dụ ghi 3 bit `[1, 1, 1]`:

```text
byte 0 = 0x11   (bit 0 ở nửa cao = 1, bit 1 ở nửa thấp = 1)
byte 1 = 0x10   (bit 2 ở nửa cao = 1, nửa thấp không dùng)
```

---

## 3. Mã thiết bị

| Thiết bị | Mã | Loại | Hệ đếm |
| --- | ---: | --- | --- |
| `SM` | `0x91` | bit | 10 |
| `SD` | `0xA9` | từ | 10 |
| `X` | `0x9C` | bit | **16** |
| `Y` | `0x9D` | bit | **16** |
| `M` | `0x90` | bit | 10 |
| `L` | `0x92` | bit | 10 |
| `F` | `0x93` | bit | 10 |
| `V` | `0x94` | bit | 10 |
| `B` | `0xA0` | bit | **16** |
| `D` | `0xA8` | từ | 10 |
| `W` | `0xB4` | từ | **16** |
| `R` | `0xAF` | từ | 10 |
| `ZR` | `0xB0` | từ | 10 |
| `TN` | `0xC2` | từ | 10 |
| `CN` | `0xC5` | từ | 10 |
| `TS` | `0xC1` | bit | 10 |
| `CS` | `0xC4` | bit | 10 |
| `Z` | `0xCC` | từ | 10 |

**X, Y, B, W đánh số theo hệ MƯỜI SÁU.** `X10` là 16, không phải 10. Đây là
chỗ dễ sai nhất khi gõ tay địa chỉ, và sai một đơn vị ở đây nghĩa là đọc nhầm
đầu vào.

---

## 4. Mã kết thúc hay gặp

| Mã | Nghĩa | Cách chữa |
| --- | --- | --- |
| `0x0000` | bình thường | |
| `0xC050` | định dạng dữ liệu sai | ASCII/binary lẫn lộn |
| `0xC051` | PC No. sai | phải là `FF` |
| `0xC052` | Module I/O No. sai | phải là `03FF` |
| `0xC054` | độ dài dữ liệu sai | xem mục 1 |
| `0xC058` | command/subcommand sai | PLC không hỗ trợ lệnh này |
| `0xC059` | subcommand sai | sai loại thiết bị (bit vs từ) |
| `0xC05B` | số hiệu thiết bị sai | thiết bị không tồn tại trên PLC này |
| `0xC05D` | số điểm vượt giới hạn | đọc ít hơn mỗi lần |
| `0xC200` | mật khẩu truy cập từ xa sai | |
| `0xC810` | lỗi truyền thông | |
| `0xC811` | hết thời gian chờ | tăng monitoring timer |

---

## 5. Cài đặt trên PLC (GX Works3, FX5U)

1. **Navigation → Parameter → FX5UCPU → Module Parameter → Ethernet Port**
2. **Own Node Settings**: đặt IP cho PLC, ví dụ `192.168.1.10`
3. **Object Device Settings → SLMP**:
   - Protocol: **TCP**
   - Port: **5007** (hoặc cổng khác, phải khớp `PlcConfig.port`)
   - Frame: **3E binary**  ← quan trọng, `mc.py` chỉ nói 3E binary
4. Nếu PLC có mật khẩu truy cập từ xa thì phải gửi lệnh mở khoá trước — `mc.py`
   chưa cài đặt phần này.

Kiểm tra nhanh từ PC:

```powershell
Test-NetConnection 192.168.1.10 -Port 5007
```

---

## 6. Tự kiểm tra

```bash
python -m vision_qc check
```

Mục **Khung tin MC** so từng byte của khung đọc `D100` với bảng ở mục 1. Mục
**MC qua TCP** dựng một PLC giả (`SimMcPlc`) nói đúng giao thức này trong tiến
trình rồi chạy 16 lượt đọc/ghi thật qua socket.

Cách kiểm tra này có chủ đích: chỉ thử vòng tròn gửi-rồi-đọc-lại là **không
đủ**, vì nếu cả bên gửi lẫn bên nhận cùng hiểu sai một chỗ thì vòng tròn vẫn
"đạt". Phải so với tài liệu thì mới bắt được lỗi.
