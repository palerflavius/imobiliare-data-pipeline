from curl_cffi import requests
from curl_cffi.requests.exceptions import RequestException
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from scraper.core.config import BROWSER_IMPERSONATE, HEADERS, HTTP_PROXY


def create_client() -> requests.Session:
    """Create a persistent browser-impersonating HTTP session."""
    proxies = {"http": HTTP_PROXY, "https": HTTP_PROXY} if HTTP_PROXY else None
    return requests.Session(
        headers=HEADERS,
        impersonate=BROWSER_IMPERSONATE,
        proxies=proxies,
    )


def is_retryable_fetch_error(error: BaseException) -> bool:
    """Retry transient network failures, rate limits, and server errors."""
    if isinstance(error, RequestException):
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            return True
        return status_code == 429 or status_code >= 500

    return False


@retry(
    retry=retry_if_exception(is_retryable_fetch_error),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True,
)
def fetch_response(client: requests.Session, url: str):
    """Fetch a URL with shared headers and retry handling."""
    response = client.get(url, timeout=30, allow_redirects=True)
    response.raise_for_status()
    return response


def fetch(client: requests.Session, url: str) -> str:
    """Return only the response body for scraper pages."""
    response = fetch_response(client, url)
    return response.text
