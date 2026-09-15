"""
mc.py — Giao thức MC (SLMP) của Mitsubishi, viết từ đầu trên socket TCP.

PLC Mitsubishi FX5U / Q series nói giao thức này. Không cần cài thư viện nào:
khung tin là một chuỗi byte cố định, gửi qua TCP, đọc trả lời. Hết.

Khung 3E (binary) — yêu cầu:

    offset  nội dung                      giá trị
    ------  ---------------------------   -----------------------
      0     subheader (2 byte)            50 00
      2     network no (1)                00
      3     PC no (1)                     FF
      4     module I/O no (2, LE)         FF 03   (= 0x03FF)
      6     module station no (1)         00
      7     độ dài dữ liệu (2, LE)        tuỳ
      9     monitoring timer (2, LE)      10 00   (= 4 giây)
     11     command (2, LE)               01 04   (đọc hàng loạt)
     13     subcommand (2, LE)            00 00   (theo từ) / 01 00 (theo bit)
     15     dữ liệu                       tuỳ lệnh

Trả lời giống hệt, chỉ khác subheader `D0 00` và chèn 2 byte mã kết thúc ở
offset 9; dữ liệu bắt đầu từ offset 11.

    đọc theo từ  : mã thiết bị (1) + đầu thiết bị (3, LE) + số điểm (2, LE)
    ghi theo từ  : như trên, rồi tới các giá trị (2 byte/từ, LE)
    theo bit     : như trên, nhưng dữ liệu ĐÓNG GÓI 2 bit vào 1 byte,
                   bit thứ nhất nằm ở nửa CAO của byte.
"""

from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass, field

# ---------------------------------------------------------------------
# Hằng số giao thức
# ---------------------------------------------------------------------

SUBHEADER_REQUEST = b"\x50\x00"
SUBHEADER_RESPONSE = b"\xd0\x00"

# (network no, PC no, module I/O no, module station no)
ROUTE = b"\x00\xff\xff\x03\x00"

MONITORING_TIMER = 0x0010  # 16 x 250 ms = 4 giây

CMD_BATCH_READ = 0x0401
CMD_BATCH_WRITE = 0x1401

SUB_WORD = 0x0000
SUB_BIT = 0x0001

HEADER_REQUEST_LEN = 15   # trước phần dữ liệu
HEADER_RESPONSE_LEN = 11  # trước phần dữ liệu (đã tính 2 byte mã kết thúc)

# ---------------------------------------------------------------------
# Mã thiết bị
# ---------------------------------------------------------------------

DEVICE_CODES: dict[str, int] = {
    "SM": 0x91,   # relay đặc biệt (bit)
    "SD": 0xA9,   # thanh ghi đặc biệt (từ)
    "X": 0x9C,    # relay vào (bit)
    "Y": 0x9D,    # relay ra (bit)
    "M": 0x90,    # relay nội (bit)
    "L": 0x92,    # relay chốt (bit)
    "F": 0x93,    # annunciator (bit)
    "V": 0x94,    # relay cạnh (bit)
    "B": 0xA0,    # relay link (bit)
    "D": 0xA8,    # thanh ghi dữ liệu (từ)
    "W": 0xB4,    # thanh ghi link (từ)
    "R": 0xAF,    # thanh ghi file (từ)
    "ZR": 0xB0,   # thanh ghi file, iQ-R / iQ-F (từ)
    "TN": 0xC2,   # giá trị hiện tại timer (từ)
    "CN": 0xC5,   # giá trị hiện tại counter (từ)
    "TS": 0xC1,   # tiếp điểm timer (bit)
    "CS": 0xC4,   # tiếp điểm counter (bit)
    "Z": 0xCC,    # thanh ghi chỉ số (từ)
}

# X, Y, B, W đánh số theo hệ MƯỜI SÁU, không phải thập phân.
# Đây là chỗ dễ sai nhất khi viết tay địa chỉ: X10 là 16, không phải 10.
HEX_DEVICES = frozenset({"X", "Y", "B", "W"})

# Mã kết thúc — chỉ liệt kê những cái hay gặp, kèm cách chữa.
END_CODES: dict[int, str] = {
    0x0000: "bình thường",
    0xC050: "định dạng dữ liệu sai (ASCII/binary lẫn lộn)",
    0xC051: "số hiệu PC sai",
    0xC052: "số hiệu module I/O sai",
    0xC053: "số hiệu station sai",
    0xC054: "độ dài dữ liệu yêu cầu sai",
    0xC056: "dữ liệu yêu cầu sai",
    0xC058: "command/subcommand sai",
    0xC059: "command/subcommand sai (thiết bị không hỗ trợ)",
    0xC05B: "số hiệu thiết bị sai",
    0xC05C: "dữ liệu yêu cầu sai (số điểm)",
    0xC05D: "số điểm vượt quá giới hạn",
    0xC05F: "dữ liệu yêu cầu sai",
    0xC060: "đầu thiết bị sai",
    0xC061: "đầu thiết bị vượt quá giới hạn",
    0xC06F: "dữ liệu yêu cầu sai",
    0xC070: "số điểm sai (không xử lý được)",
    0xC0B5: "dữ liệu yêu cầu sai",
    0xC200: "mật khẩu truy cập từ xa không đúng",
    0xC201: "mật khẩu truy cập từ xa đang bị khoá",
    0xC810: "lỗi truyền thông",
    0xC811: "hết thời gian chờ truyền thông",
}


class McError(RuntimeError):
    """PLC trả về mã kết thúc khác 0, hoặc khung tin không đọc được."""


# ---------------------------------------------------------------------
# Địa chỉ thiết bị
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Device:
    """Một địa chỉ PLC, ví dụ `D100`, `M200`, `ZR512`, `X1F`."""

    name: str   # "D", "M", "ZR", ...
    number: int

    @property
    def code(self) -> int:
        try:
            return DEVICE_CODES[self.name]
        except KeyError:
            raise McError(
                f"Không biết thiết bị {self.name!r}. "
                f"Có: {', '.join(sorted(DEVICE_CODES))}"
            ) from None

    @property
    def is_bit(self) -> bool:
        """Thiết bị bit hay thiết bị từ? Quyết định subcommand."""
        return self.name in {"X", "Y", "M", "L", "F", "V", "B", "SM", "TS", "CS"}

    def __str__(self) -> str:
        return f"{self.name}{self.number}"


def parse_device(text: str) -> Device:
    """`"D100"` -> `Device("D", 100)`. Tự nhận hệ đếm theo loại thiết bị."""
    if not text:
        raise McError("Địa chỉ thiết bị rỗng")

    upper = text.strip().upper()
    split = 0
    while split < len(upper) and upper[split].isalpha():
        split += 1

    name, digits = upper[:split], upper[split:]
    if not name or not digits:
        raise McError(f"Địa chỉ thiết bị không hợp lệ: {text!r}")

    base = 16 if name in HEX_DEVICES else 10
    try:
        number = int(digits, base)
    except ValueError:
        kind = "thập lục phân" if base == 16 else "thập phân"
        raise McError(
            f"{text!r}: phần số {digits!r} không phải số {kind} "
            f"(thiết bị {name} đánh số theo hệ {'16' if base == 16 else '10'})"
        ) from None

    if number > 0xFFFFFF:
        raise McError(f"{text!r}: số hiệu thiết bị quá lớn (tối đa 16777215)")

    return Device(name, number)


# ---------------------------------------------------------------------
# Đóng gói / bóc gói
# ---------------------------------------------------------------------


def build_request(command: int, subcommand: int, data: bytes = b"") -> bytes:
    """Ghép một khung yêu cầu 3E hoàn chỉnh."""
    body = struct.pack("<HH", MONITORING_TIMER, command) + struct.pack("<H", subcommand) + data
    header = SUBHEADER_REQUEST + ROUTE + struct.pack("<H", len(body))
    return header + body


def _device_bytes(device: Device, points: int) -> bytes:
    if points <= 0:
        raise McError(f"số điểm phải lớn hơn 0, đang là {points}")
    return (
        bytes([device.code])
        + struct.pack("<I", device.number)[:3]   # đầu thiết bị: 3 byte
        + struct.pack("<H", points)
    )


def encode_read(device: Device, points: int) -> tuple[int, bytes]:
    """(subcommand, dữ liệu) cho lệnh đọc hàng loạt."""
    sub = SUB_BIT if device.is_bit else SUB_WORD
    return sub, _device_bytes(device, points)


def encode_write(device: Device, values: list[int] | list[bool]) -> tuple[int, bytes]:
    """(subcommand, dữ liệu) cho lệnh ghi hàng loạt."""
    if not values:
        raise McError("không có giá trị nào để ghi")

    points = len(values)

    if device.is_bit:
        # Theo bit: 2 điểm gói vào 1 byte, điểm thứ nhất ở nửa CAO.
        packed = bytearray((points + 1) // 2)
        for index, value in enumerate(values):
            if value:
                packed[index // 2] |= 0x10 >> (4 * (index % 2))
        return SUB_BIT, _device_bytes(device, points) + bytes(packed)

    words = b"".join(struct.pack("<H", int(v) & 0xFFFF) for v in values)
    return SUB_WORD, _device_bytes(device, points) + words


@dataclass
class McResponse:
    end_code: int
    data: bytes = b""
    raw: bytes = field(default=b"", repr=False)

    @property
    def ok(self) -> bool:
        return self.end_code == 0

    def describe(self) -> str:
        meaning = END_CODES.get(self.end_code, "mã lạ — tra tài liệu SH-080008")
        return f"0x{self.end_code:04X}: {meaning}"


def parse_response(raw: bytes) -> McResponse:
    """Bóc khung trả lời 3E."""
    if len(raw) < HEADER_RESPONSE_LEN:
        raise McError(
            f"Trả lời quá ngắn: {len(raw)} byte, cần ít nhất {HEADER_RESPONSE_LEN}. "
            f"Nhận được: {raw.hex(' ')}"
        )
    if raw[:2] != SUBHEADER_RESPONSE:
        raise McError(
            f"Subheader sai: nhận {raw[:2].hex(' ')}, cần d0 00. "
            f"Có phải PLC trả về khung 1E/2E/4E không? (kiểm tra cài đặt SLMP)"
        )

    declared = struct.unpack_from("<H", raw, 7)[0]
    end_code = struct.unpack_from("<H", raw, 9)[0]
    data = raw[HEADER_RESPONSE_LEN:]

    # Độ dài khai báo tính từ monitoring timer tới hết dữ liệu, tức là
    # 2 (end code) + len(data) trong khung trả lời.
    if declared != len(data) + 2:
        raise McError(
            f"Độ dài khai báo {declared} không khớp thực tế {len(data) + 2} "
            f"(nhận {len(raw)} byte)"
        )

    return McResponse(end_code=end_code, data=data, raw=raw)


def decode_words(data: bytes) -> list[int]:
    """Dữ liệu theo từ -> danh sách số nguyên có dấu (PLC dùng bù hai)."""
    if len(data) % 2:
        raise McError(f"dữ liệu theo từ phải chẵn byte, đang có {len(data)}")
    count = len(data) // 2
    return list(struct.unpack(f"<{count}h", data))


def decode_bits(data: bytes, points: int) -> list[int]:
    """Dữ liệu theo bit -> danh sách 0/1, giải nén từ 2 bit mỗi byte."""
    if len(data) < (points + 1) // 2:
        raise McError(
            f"cần {(points + 1) // 2} byte cho {points} bit, chỉ có {len(data)}"
        )
    out: list[int] = []
    for index in range(points):
        byte = data[index // 2]
        out.append((byte >> (4 if index % 2 == 0 else 0)) & 0x0F)
    return out


# ---------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------


class McClient:
    """Nói chuyện với PLC Mitsubishi qua TCP."""

    def __init__(self, host: str, port: int = 5007, timeout_s: float = 2.0, retries: int = 2):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.retries = retries
        self.sock: socket.socket | None = None

    # -- kết nối ------------------------------------------------------

    def connect(self) -> None:
        if self.sock is not None:
            return
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        except OSError as exc:
            raise McError(
                f"Không kết nối được PLC {self.host}:{self.port} — {exc}\n"
                f"  - PLC có bật SLMP qua Ethernet không? (GX Works3: "
                f"Module Parameter > Ethernet Port > SLMP)\n"
                f"  - Cổng mặc định thường là 5007; kiểm tra lại trong cấu hình PLC.\n"
                f"  - PC và PLC có cùng dải mạng không? Thử `ping {self.host}`."
            ) from exc

        # TCP không có ranh giới bản tin, nên phải tự biết cần đọc bao nhiêu
        # byte. Nagle làm chậm gói nhỏ, tắt đi cho nhanh.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.timeout_s)
        self.sock = sock

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def __enter__(self) -> McClient:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- gửi / nhận ---------------------------------------------------

    def _recv_exactly(self, count: int) -> bytes:
        assert self.sock is not None
        chunks: list[bytes] = []
        got = 0
        while got < count:
            chunk = self.sock.recv(count - got)
            if not chunk:
                raise McError("PLC đóng kết nối giữa chừng")
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def request(self, command: int, subcommand: int, data: bytes = b"") -> McResponse:
        """Gửi một khung và đọc trả lời. Tự thử lại khi đứt kết nối."""
        last: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                self.connect()
                assert self.sock is not None

                self.sock.sendall(build_request(command, subcommand, data))

                # Đọc 9 byte đầu để biết độ dài, rồi đọc nốt phần còn lại.
                head = self._recv_exactly(9)
                declared = struct.unpack_from("<H", head, 7)[0]
                rest = self._recv_exactly(declared)
                return parse_response(head + rest)

            except (OSError, McError) as exc:
                last = exc
                self.close()
                if attempt < self.retries:
                    continue

        raise McError(f"Gửi tới PLC {self.host}:{self.port} thất bại sau "
                      f"{self.retries + 1} lần: {last}") from last

    # -- API tiện dụng ------------------------------------------------

    def read_words(self, address: str, points: int = 1) -> list[int]:
        device = parse_device(address)
        sub, data = encode_read(device, points)
        response = self.request(CMD_BATCH_READ, sub, data)
        if not response.ok:
            raise McError(f"Đọc {address} ({points} từ) lỗi — {response.describe()}")
        return decode_words(response.data)

    def read_word(self, address: str) -> int:
        return self.read_words(address, 1)[0]

    def write_words(self, address: str, values: list[int]) -> None:
        device = parse_device(address)
        sub, data = encode_write(device, values)
        response = self.request(CMD_BATCH_WRITE, sub, data)
        if not response.ok:
            raise McError(f"Ghi {address} lỗi — {response.describe()}")

    def write_word(self, address: str, value: int) -> None:
        self.write_words(address, [value])

    def read_bits(self, address: str, points: int = 1) -> list[int]:
        device = parse_device(address)
        sub, data = encode_read(device, points)
        response = self.request(CMD_BATCH_READ, sub, data)
        if not response.ok:
            raise McError(f"Đọc {address} ({points} bit) lỗi — {response.describe()}")
        return decode_bits(response.data, points)

    def read_bit(self, address: str) -> bool:
        return bool(self.read_bits(address, 1)[0])

    def write_bits(self, address: str, values: list[int]) -> None:
        device = parse_device(address)
        sub, data = encode_write(device, values)
        response = self.request(CMD_BATCH_WRITE, sub, data)
        if not response.ok:
            raise McError(f"Ghi {address} lỗi — {response.describe()}")

    def write_bit(self, address: str, value: bool) -> None:
        self.write_bits(address, [1 if value else 0])


# ---------------------------------------------------------------------
# PLC giả — nói đúng giao thức đó, để thử mà không cần phần cứng
# ---------------------------------------------------------------------


class SimMcPlc:
    """Một PLC Mitsubishi tí hon chạy trong tiến trình, phục vụ qua TCP thật.

    Không phải đồ chơi cho vui: nó là thứ duy nhất cho phép thử `McClient`
    và `station` end-to-end khi chưa có PLC trong tay. Nó cũng là thứ chứng
    minh khung tin đúng — nếu bên gửi và bên nhận cùng hiểu sai một chỗ thì
    test sẽ không phát hiện ra, nên `tests/test_mc.py` còn kiểm tra thêm
    từng byte của khung so với tài liệu.

    Chạy trong luồng riêng, lắng nghe trên 127.0.0.1 với cổng tự chọn.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.host = host
        self.port = port
        self.words: dict[tuple[str, int], int] = {}
        self.bits: dict[tuple[str, int], int] = {}
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.requests_served = 0
        self.lock = threading.Lock()

    # -- vòng đời -----------------------------------------------------

    def start(self) -> SimMcPlc:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._server.settimeout(0.2)

        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._server is not None:
            self._server.close()
            self._server = None

    def __enter__(self) -> SimMcPlc:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- trạng thái giả lập -------------------------------------------

    def set_word(self, address: str, value: int) -> None:
        device = parse_device(address)
        with self.lock:
            self.words[(device.name, device.number)] = value & 0xFFFF

    def get_word(self, address: str) -> int:
        device = parse_device(address)
        with self.lock:
            return self.words.get((device.name, device.number), 0)

    def set_bit(self, address: str, value: bool) -> None:
        device = parse_device(address)
        with self.lock:
            self.bits[(device.name, device.number)] = 1 if value else 0

    def get_bit(self, address: str) -> bool:
        device = parse_device(address)
        with self.lock:
            return bool(self.bits.get((device.name, device.number), 0))

    # -- phục vụ ------------------------------------------------------

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                assert self._server is not None
                conn, _ = self._server.accept()
            except (TimeoutError, OSError):
                continue
            try:
                self._handle(conn)
            except OSError:
                pass
            finally:
                conn.close()

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(0.5)
        while not self._stop.is_set():
            head = self._recv(conn, HEADER_REQUEST_LEN)
            if head is None:
                return

            # `request data length` đếm từ monitoring timer (offset 9), mà
            # 15 byte đầu đã bao gồm 6 byte đó (timer + command + subcommand).
            # Nên phần còn thiếu là declared - 6, KHÔNG phải declared.
            declared = struct.unpack_from("<H", head, 7)[0]
            remaining = declared - 6
            tail = self._recv(conn, remaining) if remaining > 0 else b""
            if tail is None:
                return

            frame = head + tail
            conn.sendall(self._process(frame))

    @staticmethod
    def _recv(conn: socket.socket, count: int) -> bytes | None:
        chunks: list[bytes] = []
        got = 0
        while got < count:
            try:
                chunk = conn.recv(count - got)
            except (TimeoutError, OSError):
                if not chunks:
                    return None
                raise
            if not chunk:
                return None
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def _respond(self, end_code: int, data: bytes = b"") -> bytes:
        body = struct.pack("<H", end_code) + data
        return SUBHEADER_RESPONSE + ROUTE + struct.pack("<H", len(body)) + body

    def _process(self, frame: bytes) -> bytes:
        if frame[:2] != SUBHEADER_REQUEST:
            return self._respond(0xC050)

        command, subcommand = struct.unpack_from("<HH", frame, 11)
        payload = frame[HEADER_REQUEST_LEN:]
        self.requests_served += 1

        if command not in (CMD_BATCH_READ, CMD_BATCH_WRITE):
            return self._respond(0xC058)
        if subcommand not in (SUB_WORD, SUB_BIT):
            return self._respond(0xC059)
        if len(payload) < 6:
            return self._respond(0xC054)

        code = payload[0]
        number = int.from_bytes(payload[1:4], "little")
        points = struct.unpack_from("<H", payload, 4)[0]

        name = next((n for n, c in DEVICE_CODES.items() if c == code), None)
        if name is None:
            return self._respond(0xC05B)
        if points == 0:
            return self._respond(0xC05D)

        if command == CMD_BATCH_READ:
            if subcommand == SUB_WORD:
                with self.lock:
                    values = [self.words.get((name, number + i), 0) for i in range(points)]
                return self._respond(0, struct.pack(f"<{points}H", *values))

            with self.lock:
                bits = [self.bits.get((name, number + i), 0) for i in range(points)]
            packed = bytearray((points + 1) // 2)
            for index, bit in enumerate(bits):
                if bit:
                    packed[index // 2] |= 0x10 >> (4 * (index % 2))
            return self._respond(0, bytes(packed))

        # ghi
        if subcommand == SUB_WORD:
            need = points * 2
            if len(payload) < 6 + need:
                return self._respond(0xC054)
            values = struct.unpack_from(f"<{points}H", payload, 6)
            with self.lock:
                for i, value in enumerate(values):
                    self.words[(name, number + i)] = value
            return self._respond(0)

        need = (points + 1) // 2
        if len(payload) < 6 + need:
            return self._respond(0xC054)
        packed = payload[6:6 + need]
        with self.lock:
            for index in range(points):
                bit = (packed[index // 2] >> (4 if index % 2 == 0 else 0)) & 0x0F
                self.bits[(name, number + index)] = bit
        return self._respond(0)

    # -- tiện cho test ------------------------------------------------

    @property
    def addresses(self) -> dict[str, int]:
        """Toàn bộ trạng thái, dạng đọc được."""
        out: dict[str, int] = {}
        with self.lock:
            out.update({f"{n}{i}": v for (n, i), v in self.words.items()})
            out.update({f"{n}{i}": v for (n, i), v in self.bits.items()})
        return out
