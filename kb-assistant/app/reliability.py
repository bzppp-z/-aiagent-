"""Day 31-32：可靠性与成本控制组件（与业务解耦，便于单测）

三个组件：
  ConcurrencyLimiter —— 并发闸门（满了立即拒绝，而不是排队拖死）
  SimpleCache        —— LRU 问答缓存（相同问题秒回）
  TokenBudget        —— 按天 Token 预算 + 成本估算（超限熔断）
"""
import threading
import time
from collections import OrderedDict


class ConcurrencyLimiter:
    """并发闸门：try_acquire 不等待，拿不到立刻返回 False（HTTP 层回 429）。"""

    def __init__(self, limit: int = 4):
        self.limit = limit
        self._sem = threading.BoundedSemaphore(limit)
        self.rejected = 0

    def try_acquire(self) -> bool:
        ok = self._sem.acquire(blocking=False)
        if not ok:
            self.rejected += 1
        return ok

    def release(self) -> None:
        try:
            self._sem.release()
        except ValueError:
            pass


class SimpleCache:
    """LRU 缓存：question -> (answer, sources, usage)。线程安全，超量淘汰最旧。"""

    def __init__(self, maxsize: int = 128):
        self.maxsize = maxsize
        self._d: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
            self.misses += 1
            return None

    def set(self, key: str, value: tuple) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)


class TokenBudget:
    """按天 Token 预算与成本估算（DeepSeek 价位为估算值，可用环境变量改）。"""

    PRICE_IN_PER_M = 2.0    # 元 / 百万 输入 token
    PRICE_OUT_PER_M = 8.0   # 元 / 百万 输出 token

    def __init__(self, daily_limit: int = 200_000):
        self.limit = daily_limit
        self.used_in = 0
        self.used_out = 0
        self.day = time.strftime("%Y-%m-%d")
        self._lock = threading.Lock()

    def _roll(self) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self.day:          # 跨天自动归零
            self.day = today
            self.used_in = self.used_out = 0

    def remaining(self) -> int:
        with self._lock:
            self._roll()
            return self.limit - (self.used_in + self.used_out)

    def check(self) -> bool:
        """预算是否还够（HTTP 层在请求入口调用，不够就熔断）。"""
        return self.remaining() > 0

    def add(self, usage: dict) -> None:
        with self._lock:
            self._roll()
            self.used_in += usage.get("in", 0)
            self.used_out += usage.get("out", 0)

    def stats(self) -> dict:
        with self._lock:
            self._roll()
            cost = self.used_in / 1e6 * self.PRICE_IN_PER_M + self.used_out / 1e6 * self.PRICE_OUT_PER_M
            return {
                "day": self.day,
                "used_in": self.used_in,
                "used_out": self.used_out,
                "limit": self.limit,
                "remaining": self.limit - (self.used_in + self.used_out),
                "cost_yuan_estimate": round(cost, 4),
            }
