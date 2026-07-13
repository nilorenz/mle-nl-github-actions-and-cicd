from src.calculator import add, subtract


def test_add():
    assert add(1, 2) == 3
    assert add(-2, 2) == 0
    assert add(0, 0) == 0
    
def test_subtract():
    assert subtract(1, 2) == -1
    assert subtract(-2, 2) == -4
    assert subtract(0, 0) == 0