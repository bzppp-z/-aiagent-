"""Day 9 + Day 16: 文档加载 + 切片（两种模式）

用法:
  python chunking.py          # fix  模式: 固定长度+重叠切片 → output/   (Day 9 原版, 留作对比)
  python chunking.py struct   # struct 模式: 剥元信息 + 按标题结构切 → output_struct/ (Day 16 新版)

链路位置: 文档 → [加载] → [切片] → (Day 10 向量化) → ...
"""
import os
import re
import sys
import json


def load_documents_multi(root=".", config="corpus_sources.json"):
    """Day 19+: 多源语料加载 —— 读 corpus_sources.json, 汇总所有来源。

    企业知识库的常见形态: 知识源外挂、配置化接入。
    来源分两类:
      dirs  —— 目录, 取其中全部 .md/.txt
      files —— 单文件清单
    路径不存在时打警告并跳过, 不让整个建库崩掉。
    """
    import json
    docs, missing = [], []
    with open(os.path.join(root, config), encoding="utf-8") as f:
        cfg = json.load(f)
    sources = cfg["sources"]
    kinds = cfg.get("kinds", {})   # kind: course=课程知识 / log=学员日记（默认不参与检索）

    def add(path):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as fp:
            docs.append((name, fp.read(), kinds.get(name, "course")))

    for d in sources.get("dirs", []):
        d = os.path.join(root, d)
        if not os.path.isdir(d):
            missing.append(d)
            continue
        for name in os.listdir(d):
            if name.endswith((".md", ".txt")):
                add(os.path.join(d, name))
    for p in sources.get("files", []):
        p = os.path.join(root, p)
        if os.path.isfile(p):
            add(p)
        else:
            missing.append(p)

    for m in missing:
        print(f"  [警告] 语料源缺失, 已跳过: {m}")
    return docs


def load_documents(folder):
    """单目录加载（保留: 单源场景 / 兼容旧用法）。"""
    docs = []
    for name in os.listdir(folder):
        if name.endswith((".md", ".txt")):
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as f:
                docs.append((name, f.read()))
    return docs


def strip_frontmatter(text):
    """Day 16 清洗: 剥掉 Markdown 开头的元信息块（title/description/keywords/tags）。

    这些块是 BC-1 里垃圾 chunk 的源头——关键词密度高，向量离什么问题都近，
    还占检索名额。它们是给网站/渲染用的配置，不是给读者看的内容。
    """
    if text.startswith("---"):
        end = text.find("---", 3)          # 找配对的收尾 ---
        if end != -1:
            return text[end + 3 :].lstrip("\n")
    return text


def chunk_text(text, chunk_size=300, overlap=50):
    """固定长度 + 重叠切片（Day 9 原版，fix 模式仍在用它做超长小节的兜底）。

    chunk_size: 每片最多多少字符
    overlap:    相邻两片重叠多少字符
    步长 = chunk_size - overlap（滑动窗口每次向右挪这么多）
    """
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size，否则窗口永远不前进")

    chunks = []
    i = 0
    while i < len(text):
        piece = text[i : i + chunk_size].strip()
        if piece:  # 整段都是空白的片跳过，防止死循环
            chunks.append({"text": piece})
        i += chunk_size - overlap
    return chunks


def chunk_structured(text, max_chars=500, overlap=100):
    """Day 16: 结构感知切分 —— 按 Markdown 标题切，一个小节一片。

    为什么它比固定长度好（对照 BC-1）:
    1. 边界天然落在话题转换处（标题就是作者标注的话题边界），
       不会再出现「经验值段落和 chunk太害处的论述被焊在同一片」；
    2. 每片开头拼上标题路径（如 切片策略 > 策略一），检索命中自带上下文，
       标题关键词也参与编码，提高召回；
    3. 小节超过 max_chars 时，退回固定长度+重叠细分兜底，保证不会超 Embedding 上限。
    """
    sections = []          # [(标题路径列表, 小节全文)]
    cur_head, cur = [], []  # 当前小节的标题路径 / 正文行

    for line in text.split("\n"):
        m = re.match(r"^(#{1,4})\s+(.*)", line)   # 匹配 # 到 #### 的标题行
        if m:
            if cur:
                sections.append((cur_head, "\n".join(cur)))
            level = len(m.group(1))               # # 是 1 级, ## 是 2 级...
            # 标题路径: 2 级标题替换掉路径末尾, 3 级追加 —— 保持层级感
            cur_head = cur_head[: level - 1] + [m.group(2).strip()]
            cur = [line]                          # 标题行本身也留在正文里
        else:
            cur.append(line)
    if cur:
        sections.append((cur_head, "\n".join(cur)))

    chunks = []
    for head, body in sections:
        prefix = (" > ".join(head) + "\n") if head else ""
        body = body.strip()
        if not body:
            continue
        if len(body) <= max_chars:
            chunks.append({"text": prefix + body})       # 一整个小节一片
        else:
            for sub in chunk_text(body, max_chars, overlap):   # 超长小节兜底细分
                chunks.append({"text": prefix + sub["text"]})
    return chunks


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "fix"   # fix=Day9 原版 / struct=Day16 新版

    docs = load_documents_multi()
    print(f"  语料源共 {len(docs)} 份文档")
    if not docs:
        print("没有加载到任何语料，检查 corpus_sources.json")
        return

    if mode == "fix":
        out_dir, tag, sizes = "index/chunks_fix", "chunk", (100, 300)
        os.makedirs(out_dir, exist_ok=True)
        print(f"[fix 模式] 固定长度+重叠切片 → {out_dir}/")
        for name, text, kind in docs:
            stem = os.path.splitext(name)[0]
            for size in sizes:
                overlap = size // 5
                chunks = chunk_text(text, chunk_size=size, overlap=overlap)
                out_path = os.path.join(out_dir, f"{stem}_{tag}{size}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"source": name, "kind": kind, "chunk_size": size, "overlap": overlap, "chunks": chunks},
                        f, ensure_ascii=False, indent=2,
                    )
                print(f"  {name} | chunk_size={size}, overlap={overlap} → {len(chunks)} 片")
    elif mode == "struct":
        out_dir, max_chars, overlap = "index/chunks_struct", 500, 100
        os.makedirs(out_dir, exist_ok=True)
        print(f"[struct 模式] 剥元信息 + 按标题切 → {out_dir}/")
        for name, text, kind in docs:
            stem = os.path.splitext(name)[0]
            clean = strip_frontmatter(text)
            chunks = chunk_structured(clean, max_chars=max_chars, overlap=overlap)
            out_path = os.path.join(out_dir, f"{stem}_struct.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"source": name, "kind": kind, "chunk_size": max_chars, "overlap": overlap, "chunks": chunks},
                    f, ensure_ascii=False, indent=2,
                )
            print(f"  {name} | {len(chunks)} 片 (对比 fix 模式 chunk300)")
    else:
        print(f"未知模式: {mode}，用 fix 或 struct")


if __name__ == "__main__":
    main()
