"""Bounded, detached diagnostic writes outside the vehicle control loop."""
import threading
import time
from collections import OrderedDict


class DiagnosticWriter(object):
    def __init__(self, write, capacity=16):
        if not callable(write) or type(capacity) is not int or capacity < 2:
            raise ValueError('invalid diagnostic writer')
        self.write, self.capacity = write, capacity
        self.condition = threading.Condition()
        self.pending = OrderedDict()
        self.thread, self.busy, self.closed = None, False, False
        self.written = self.dropped = self.failed = 0

    def submit(self, path, detached_data):
        with self.condition:
            if self.closed:
                return False
            if path in self.pending:
                del self.pending[path]
                self.dropped += 1
            elif len(self.pending) >= self.capacity:
                self.pending.popitem(last=False)
                self.dropped += 1
            self.pending[path] = detached_data
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='captain-diagnostics')
                self.thread.daemon = True
                self.thread.start()
            self.condition.notify_all()
        return True

    def _run(self):
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if not self.pending and self.closed:
                    return
                path, data = self.pending.popitem(last=False)
                self.busy = True
            try:
                ok = self.write(path, data)
            except Exception:
                ok = False
            with self.condition:
                self.written += int(ok is not False)
                self.failed += int(ok is False)
                self.busy = False
                self.condition.notify_all()

    def flush(self, timeout=2.0):
        deadline = time.perf_counter()+timeout
        with self.condition:
            while self.pending or self.busy:
                remaining = deadline-time.perf_counter()
                if remaining <= 0:
                    return False
                self.condition.wait(remaining)
        return True

    def close(self, timeout=2.0):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout)
            return not self.thread.is_alive()
        return True

    def status(self):
        with self.condition:
            return dict(pending=len(self.pending), busy=self.busy, written=self.written,
                        dropped=self.dropped, failed=self.failed, capacity=self.capacity)
