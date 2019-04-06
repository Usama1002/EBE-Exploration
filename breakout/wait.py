import sys

def wait(s):
    if sys.version_info[0] < 3:
        _ = raw_input(s)
    else:
        _ = input(s)

