"""MySQL 数据层：材料查询 + ChromaDB 检索失败时的 Fallback 查询。"""

import os

import pymysql
from dotenv import load_dotenv

from fault_tolerance import (
    QueryConnectionError,
    QueryDataError,
    fallback_to_mysql,
    retry_with_timeout,
)

load_dotenv()


def _connect():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "mattress"),
        charset="utf8mb4",
        connect_timeout=5,
        read_timeout=10,
    )


def query_materials(category=None, preference=None):
    conn = _connect()
    cursor = conn.cursor()
    sql = "SELECT name, price, preference, description FROM materials WHERE 1=1"
    params = []
    if category:
        sql += " AND category=%s"
        params.append(category)
    if preference:
        sql += " AND preference=%s"
        params.append(preference)
    cursor.execute(sql, params)
    results = cursor.fetchall()
    conn.close()
    if not results:
        raise QueryDataError("没有符合条件的材料")
    return results


def search_docs_fallback_mysql(question):
    """ChromaDB 检索失败时的 Fallback：用 MySQL 按关键词模糊查。"""
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, description FROM materials WHERE name LIKE %s OR description LIKE %s",
        (f"%{question}%", f"%{question}%"),
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    # 正常查询
    print("=== 正常查询 ===")
    print(query_materials("填充", "偏软"))

    # 带重试和超时的查询
    print("\n=== 带重试和超时的查询 ===")
    result = retry_with_timeout(query_materials, ("填充", "偏软"), max_retries=3, timeout=10)
    print(result)

    # Fallback：ChromaDB 失败时退回 MySQL
    print("\n=== Fallback 测试 ===")

    def fake_chroma(question):
        raise QueryConnectionError("ChromaDB 未启动")

    result = fallback_to_mysql(fake_chroma, search_docs_fallback_mysql, ("天然新疆棉",))
    print(result)
