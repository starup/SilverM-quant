"""
DuckDB 全局连接池 - 解决并发连接配置冲突

用法:
    from database.connection import get_connection
    conn = get_connection()
    df = conn.execute("SELECT ...").fetchdf()
    conn.close()  # cursor 用完要关闭

重要: 返回的是 cursor，用完需要 close()。
"""
import threading
import duckdb
from pathlib import Path

_lock = threading.Lock()
_raw_connection = None

DB_PATH = str(Path(__file__).parent.parent / 'data' / 'Astock3.duckdb')


def get_connection(db_path: str = None, read_only: bool = False):
    """获取 DuckDB 游标（线程安全，每次返回独立 cursor）

    read_only=True 时直接返回独立只读连接的 cursor（用于多进程子进程）。
    read_only=False 时使用全局读写单例连接。
    """
    global _raw_connection
    path = db_path or DB_PATH

    if read_only:
        conn = duckdb.connect(path, read_only=True)
        return conn.cursor()

    with _lock:
        if _raw_connection is None:
            _raw_connection = duckdb.connect(path)
        else:
            try:
                _raw_connection.execute("SELECT 1")
            except Exception:
                _raw_connection = duckdb.connect(path)

    return _raw_connection.cursor()


def close_connection():
    """真正关闭全局连接（仅在进程退出时调用）"""
    global _raw_connection
    with _lock:
        if _raw_connection is not None:
            try:
                _raw_connection.close()
            except Exception:
                pass
            _raw_connection = None
