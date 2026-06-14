import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from scraper.core.config import HEADERS


def is_retryable_fetch_error(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code == 429 or status_code >= 500

    return isinstance(error, (httpx.TimeoutException, httpx.TransportError))


@retry(
    retry=retry_if_exception(is_retryable_fetch_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True,
)
def fetch_response(client: httpx.Client, url: str) -> httpx.Response:
    response = client.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response


def fetch(client: httpx.Client, url: str) -> str:
    response = fetch_response(client, url)
    return response.text
