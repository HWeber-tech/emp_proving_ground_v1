from src.core.instrument import get_all_instruments, get_instrument


def test_get_all_instruments_returns_copies() -> None:
    instruments = get_all_instruments()
    original_name = get_instrument("EURUSD").name

    instruments["EURUSD"].name = "mutated"

    assert get_instrument("EURUSD").name == original_name
