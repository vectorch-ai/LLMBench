import threading
import time
import logging


class TokenBucket:
    _lock = threading.Lock()
    _instance = None

    def __init__(self, qps: float):
        self.qps = qps
        self.refill_time = self._refill_time_generator()

    def _refill_time_generator(self):
        """Generator to provide refill times at a fixed rate."""
        t = time.time()
        avg_wait = 1.0 / self.qps
        while True:
            t += avg_wait
            yield t

    @classmethod
    def get_instance(cls, qps: float):
        """Singleton instance creation with thread safety."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(qps)
            else:
                assert cls._instance.qps == qps, "QPS value mismatch for existing instance."
            return cls._instance

    def wait_time(self):
        """Return the time to wait before the next token is available."""
        if self.qps == float("inf"):
            return 0
        
        with self._lock:
            t = next(self.refill_time)
        now = time.time()
        if now > t:
            logging.warning(f"Cannot keep up with qps: {self.qps}, delay: {now-t:.6f}s")
            return 0
        return t - now


if __name__ == "__main__":
    tb = TokenBucket.get_instance(4)
    def worker(tid):
        for i in range(10):
            wait_time = tb.wait_time()
            time.sleep(wait_time)
            print(f"{tid}: {time.time():.6f}, token {i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
