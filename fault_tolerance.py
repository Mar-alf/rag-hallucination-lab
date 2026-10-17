import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QueryTimeoutError(Exception):
    """查询超时"""
    pass


class QueryConnectionError(Exception):
    """连接失败"""
    pass


class QueryDataError(Exception):
    """数据不存在或格式错误"""
    pass


def classify_error(e):
    """错误分类：根据异常类型返回错误类别"""
    if isinstance(e, QueryTimeoutError):
        return "timeout"
    if isinstance(e, QueryConnectionError):
        return "connection"
    if isinstance(e, QueryDataError):
        return "data"
    return "unknown"


def retry_with_timeout(func, args=(), kwargs=None, max_retries=3, timeout=10):
    """
    带超时、重试、错误分类的查询封装。

    - timeout: 单次查询的超时时间（秒）
    - max_retries: 最大重试次数
    - 连接类错误：重试
    - 超时类错误：重试
    - 数据类错误：不重试，直接抛出
    """
    if kwargs is None:
        kwargs = {}

    last_error = None
    for attempt in range(1, max_retries + 1):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            if elapsed > timeout:
                raise QueryTimeoutError(f"查询耗时 {elapsed:.2f}s，超过 {timeout}s")
            return result
        except QueryDataError:
            # 数据类错误不重试
            raise
        except Exception as e:
            category = classify_error(e)
            last_error = e
            logging.warning(f"[第{attempt}次] 错误类型={category}, 原因={e}")
            if attempt < max_retries:
                time.sleep(1)
            else:
                raise

    raise last_error


def fallback_to_mysql(chroma_func, mysql_func, args=(), kwargs=None):
    """
    Fallback 机制：ChromaDB 检索失败时，退回用 MySQL 直接查。
    - chroma_func: 优先调用的函数（ChromaDB 检索）
    - mysql_func: 备用函数（MySQL 查询）
    """
    try:
        return retry_with_timeout(chroma_func, args, kwargs)
    except Exception as e:
        logging.warning(f"ChromaDB 检索失败，Fallback 到 MySQL。原因: {e}")
        return retry_with_timeout(mysql_func, args, kwargs)
