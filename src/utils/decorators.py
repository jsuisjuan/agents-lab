import functools, logging, time
from typing import Callable, TypeVar, Awaitable
from typing_extensions import ParamSpec


P = ParamSpec("P")
R = TypeVar("R")

def log_execution(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Asynchronous decorator to automatically record the start, end, 
    and execution time of each LangGraph node.
    """
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = logging.getLogger(func.__module__)
        node_name = func.__name__
        logger.info(f"START Node: '{node_name}'")
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"FINISHED Node: '{node_name}' in {elapsed:.2f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"CRASH Node: '{node_name}' after {elapsed:.2f}s | " 
                         "Error: {e}", exc_info=True)
            raise e
    return wrapper