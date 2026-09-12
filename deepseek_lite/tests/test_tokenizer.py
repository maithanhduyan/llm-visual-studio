"""Test tokenizer: chữ -> số -> chữ phải quay về đúng chỗ cũ."""

from deepseek_lite import Tokenizer


def test_roundtrip():
    tok = Tokenizer("Học mãi thì giỏi.")
    text = "Học mãi"

    assert tok.decode(tok.encode(text)) == text


def test_vocab_size():
    tok = Tokenizer("aabbcc")

    # Ba ký tự khác nhau, không phải sáu.
    assert tok.vocab_size == 3
    assert len(tok) == 3


def test_char_unknown_is_skipped():
    tok = Tokenizer("abc")

    # "z" chưa từng thấy -> bỏ qua, không làm sập chương trình.
    assert tok.encode("azb") == tok.encode("ab")


def test_ids_are_stable():
    tok = Tokenizer("abc")

    # Cùng một chữ luôn ra cùng một số.
    assert tok.encode("a") == tok.encode("a")
    assert tok.encode("abc") == [tok.stoi["a"], tok.stoi["b"], tok.stoi["c"]]


def test_save_and_load():
    tok = Tokenizer("Học mãi thì giỏi.")

    # Dựng lại từ danh sách ký tự đã lưu -> phải giống hệt.
    again = Tokenizer.load(tok.chars)

    assert again.stoi == tok.stoi
    assert again.encode("mãi") == tok.encode("mãi")
