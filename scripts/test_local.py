import sys
import pytest

def main():
    base_args = ['-vv', '--log-cli-level=DEBUG']
    
    if len(sys.argv) > 1:
        test_args = sys.argv[1:]
        pytest.main(base_args + test_args)
    else:
        pytest.main(base_args + ['test'])


if __name__ == "__main__":
    main()