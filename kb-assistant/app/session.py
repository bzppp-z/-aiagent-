"""会话管理：进程内内存 dict（v0.5 范围：单进程、重启即丢，够用不装数据库）

只存最近 HISTORY_TURNS 轮问答——历史用途有两个：
  ① 返回给前端做对话气泡；② 将来接查询改写器（v0.3 的 rag_chat 机制）。
"""
from uuid import uuid4

HISTORY_TURNS = 3


class SessionStore:
    def __init__(self):
        self._data: dict[str, list[tuple[str, str]]] = {}

    def new_id(self) -> str:
        return uuid4().hex

    def get(self, sid: str) -> list[tuple[str, str]]:
        return self._data.get(sid, [])

    def append(self, sid: str, question: str, answer: str) -> None:
        hist = self._data.setdefault(sid, [])
        hist.append((question, answer))
        del hist[:-HISTORY_TURNS]   # 只留最近 N 轮，防内存无限涨
