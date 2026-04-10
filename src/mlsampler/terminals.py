import sys
import time
import threading
from contextlib import contextmanager

def spinner(stop_event):
    symbols = ["|", "/", "-", "\\"]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rsampling... {symbols[i % len(symbols)]}")
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1

    sys.stdout.write("\rsampling completed    \n")
    sys.stdout.flush()

@contextmanager
def spinning(msg="sampling..."):
    stop_event = threading.Event()
    t = threading.Thread(target=spinner, args=(stop_event,))
    t.start()
    try:
        yield
    finally:
        stop_event.set()
        t.join()