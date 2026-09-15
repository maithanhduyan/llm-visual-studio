"""
test_mc.py — Kiểm tra giao thức MC.

Bài test quan trọng nhất ở đây là `test_khung_doc_mot_tu_d100`: nó so TỪNG BYTE
với tài liệu Mitsubishi SH-080008. Chỉ kiểm tra vòng tròn gửi-rồi-đọc lại là
không đủ — nếu cả bên gửi lẫn bên nhận cùng hiểu sai một chỗ thì vòng tròn vẫn
"đạt", và sai lầm chỉ lộ ra khi cắm vào PLC thật.
"""

from __future__ import annotations

import struct

import pytest

from vision_qc.mc import (
    CMD_BATCH_READ,
    CMD_BATCH_WRITE,
    DEVICE_CODES,
    McClient,
    McError,
    SimMcPlc,
    build_request,
    decode_bits,
    decode_words,
    encode_read,
    encode_write,
    parse_device,
    parse_response,
)

# ---------------------------------------------------------------------
# Địa chỉ thiết bị
# ---------------------------------------------------------------------


def test_doc_dia_chi_thap_phan():
    assert parse_device("D100").name == "D"
    assert parse_device("D100").number == 100
    assert parse_device("ZR512").number == 512
    assert parse_device("d100").number == 100  # không phân biệt hoa thường


def test_x_y_b_danh_so_he_16():
    """X, Y, B, W đánh số thập lục phân. X10 là 16, KHÔNG phải 10."""
    assert parse_device("X10").number == 16
    assert parse_device("Y1F").number == 31
    assert parse_device("B0A").number == 10
    assert parse_device("M10").number == 10  # M thì thập phân


def test_thiet_bi_bit_hay_tu():
    assert parse_device("M100").is_bit
    assert parse_device("X10").is_bit
    assert parse_device("Y0").is_bit
    assert not parse_device("D100").is_bit
    assert not parse_device("ZR0").is_bit


def test_dia_chi_sai_thi_bao_loi_ro_rang():
    with pytest.raises(McError, match="không hợp lệ"):
        parse_device("100")
    with pytest.raises(McError, match="thập lục phân"):
        parse_device("X1Z")
    with pytest.raises(McError, match="Không biết thiết bị"):
        unknown = parse_device("QQ10")
        _ = unknown.code


def test_ma_thiet_bi_dung_theo_tai_lieu():
    assert DEVICE_CODES["D"] == 0xA8
    assert DEVICE_CODES["M"] == 0x90
    assert DEVICE_CODES["X"] == 0x9C
    assert DEVICE_CODES["Y"] == 0x9D
    assert DEVICE_CODES["ZR"] == 0xB0
    assert DEVICE_CODES["SM"] == 0x91


# ---------------------------------------------------------------------
# Khung tin — so từng byte với tài liệu
# ---------------------------------------------------------------------


def test_khung_doc_mot_tu_d100():
    """Đọc 1 từ ở D100. Khung phải đúng 21 byte này."""
    sub, data = encode_read(parse_device("D100"), 1)
    frame = build_request(CMD_BATCH_READ, sub, data)

    expected = bytes(
        [
            0x50, 0x00,              # subheader
            0x00,                    # network no
            0xFF,                    # PC no
            0xFF, 0x03,              # module I/O no = 0x03FF
            0x00,                    # module station no
            0x0C, 0x00,              # độ dài dữ liệu = 12
            0x10, 0x00,              # monitoring timer = 4 giây
            0x01, 0x04,              # command = 0x0401 (đọc hàng loạt)
            0x00, 0x00,              # subcommand = 0x0000 (theo từ)
            0xA8,                    # mã thiết bị D
            0x64, 0x00, 0x00,        # đầu thiết bị = 100
            0x01, 0x00,              # 1 điểm
        ]
    )
    assert frame == expected
    assert len(frame) == 21
    assert frame.hex(" ") == (
        "50 00 00 ff ff 03 00 0c 00 10 00 01 04 00 00 a8 64 00 00 01 00"
    )


def test_khung_ghi_mot_tu_d200_gia_tri_1():
    sub, data = encode_write(parse_device("D200"), [1])
    frame = build_request(CMD_BATCH_WRITE, sub, data)

    assert len(frame) == 23
    assert frame.hex(" ") == (
        "50 00 00 ff ff 03 00 0e 00 10 00 01 14 00 00 a8 c8 00 00 01 00 01 00"
    )


def test_khung_ghi_hai_bit_m200():
    """Ghi bit: 2 điểm gói vào 1 byte, điểm thứ nhất ở nửa CAO."""
    sub, data = encode_write(parse_device("M200"), [1, 0])
    frame = build_request(CMD_BATCH_WRITE, sub, data)

    assert len(frame) == 22
    assert frame.hex(" ") == (
        "50 00 00 ff ff 03 00 0d 00 10 00 01 14 01 00 90 c8 00 00 02 00 10"
    )


def test_dong_goi_bit_diem_le():
    """Ghi 3 bit -> 2 byte, byte cuối chỉ dùng nửa cao."""
    sub, data = encode_write(parse_device("M0"), [1, 1, 1])
    assert sub == 0x0001
    packed = data[6:]
    assert packed == bytes([0x11, 0x10])


# ---------------------------------------------------------------------
# Bóc khung trả lời
# ---------------------------------------------------------------------


def _khung_tra_loi(end_code: int, data: bytes) -> bytes:
    body = struct.pack("<H", end_code) + data
    return b"\xd0\x00" + b"\x00\xff\xff\x03\x00" + struct.pack("<H", len(body)) + body


def test_boc_khung_tra_loi_binh_thuong():
    raw = _khung_tra_loi(0, struct.pack("<h", 1234))
    response = parse_response(raw)

    assert response.ok
    assert decode_words(response.data) == [1234]
    assert response.describe().startswith("0x0000")


def test_boc_khung_so_nguyen_co_dau():
    """PLC dùng bù hai: 0xFFFF là -1, không phải 65535."""
    raw = _khung_tra_loi(0, struct.pack("<hh", -1, -32768))
    assert decode_words(parse_response(raw).data) == [-1, -32768]


def test_ma_ket_thuc_khac_0_thi_khong_ok():
    raw = _khung_tra_loi(0xC059, b"")
    response = parse_response(raw)

    assert not response.ok
    assert response.end_code == 0xC059
    assert "command" in response.describe()


def test_subheader_sai_thi_bao_loi():
    with pytest.raises(McError, match="Subheader sai"):
        parse_response(b"\x54\x00" + b"\x00" * 20)


def test_do_dai_khai_bao_lech_thi_bao_loi():
    raw = bytearray(_khung_tra_loi(0, b"\x01\x02"))
    raw[7:9] = struct.pack("<H", 99)
    with pytest.raises(McError, match="Độ dài khai báo"):
        parse_response(bytes(raw))


def test_giai_nen_bit():
    assert decode_bits(bytes([0x10]), 2) == [1, 0]
    assert decode_bits(bytes([0x01]), 2) == [0, 1]
    assert decode_bits(bytes([0x11]), 2) == [1, 1]
    assert decode_bits(bytes([0x11, 0x10]), 3) == [1, 1, 1]


# ---------------------------------------------------------------------
# Chạy thật qua TCP — client nói chuyện với PLC giả
# ---------------------------------------------------------------------


@pytest.fixture
def plc():
    with SimMcPlc() as server:
        yield server


@pytest.fixture
def client(plc):
    with McClient(plc.host, plc.port, timeout_s=2.0) as c:
        yield c


def test_doc_ghi_tu_qua_tcp(client, plc):
    client.write_word("D100", 4242)
    assert client.read_word("D100") == 4242
    assert plc.get_word("D100") == 4242


def test_so_am_di_ve_nguyen_ven(client):
    client.write_word("D10", -1)
    assert client.read_word("D10") == -1


def test_doc_ghi_bit_qua_tcp(client, plc):
    client.write_bit("M200", True)
    client.write_bit("M201", False)
    assert client.read_bit("M200") is True
    assert client.read_bit("M201") is False
    assert plc.get_bit("M200") is True


def test_doc_nhieu_diem_mot_lan(client):
    for offset, value in enumerate([11, 22, 33, 44]):
        client.write_word(f"D{300 + offset}", value)

    assert client.read_words("D300", 4) == [11, 22, 33, 44]


def test_doc_nhieu_bit_mot_lan(client):
    for index, value in enumerate([1, 0, 1, 1, 0, 1, 0, 0, 1, 1]):
        client.write_bit(f"M{400 + index}", bool(value))

    assert client.read_bits("M400", 10) == [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]


def test_ghi_nhieu_tu_gia_tri_bien(client):
    values = [0, 1, -1, 32767, -32768]
    client.write_words("D500", values)
    assert client.read_words("D500", len(values)) == values


def test_zr_la_thiet_bi_tu(client):
    client.write_words("ZR1000", [7, 8])
    assert client.read_words("ZR1000", 2) == [7, 8]


def test_ghi_du_lieu_sai_thi_bao_loi(plc):
    with McClient(plc.host, plc.port) as client:
        with pytest.raises(McError, match="số điểm phải lớn hơn 0"):
            client.read_words("D100", 0)


def test_khong_ket_noi_duoc_thi_bao_loi_co_huong_dan():
    with pytest.raises(McError, match="SLMP"):
        McClient("127.0.0.1", 1, timeout_s=0.3, retries=0).connect()
