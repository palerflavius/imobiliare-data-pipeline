import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

from curl_cffi.requests.exceptions import RequestException
from selectolax.parser import HTMLParser

from scraper.core.config import DETAIL_REQUEST_DELAY_SECONDS, DETAIL_WORKERS, MAX_DETAIL_PAGES
from scraper.core.http_client import create_client, fetch_response
from scraper.sites.imobiliare.parser import clean_text


def find_postal_addresses(value) -> list[dict]:
    """Recursively collect PostalAddress objects from JSON-LD payloads."""
    addresses = []

    if isinstance(value, dict):
        if value.get("@type") == "PostalAddress":
            addresses.append(value)

        for child in value.values():
            addresses.extend(find_postal_addresses(child))

    if isinstance(value, list):
        for child in value:
            addresses.extend(find_postal_addresses(child))

    return addresses


def find_values_for_key(value, key: str) -> list:
    """Recursively collect every value assigned to a target JSON key."""
    values = []

    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if child_key == key:
                values.append(child_value)
            values.extend(find_values_for_key(child_value, key))

    if isinstance(value, list):
        for child in value:
            values.extend(find_values_for_key(child, key))

    return values


def json_payloads_from_scripts(tree: HTMLParser) -> list:
    """Extract candidate JSON payloads that may contain listing dates."""
    payloads = []
    decoder = json.JSONDecoder()
    target_keys = ("dateCreated", "datePublished", "dateModified")

    # The date fields can live inside embedded app state, not only JSON-LD.
    for node in tree.css("script"):
        raw_text = node.text()
        if not raw_text or not any(key in raw_text for key in target_keys):
            continue

        text = html.unescape(raw_text).strip()
        candidates = [text]
        if "\\\"" in text:
            candidates.append(text.encode("utf-8").decode("unicode_escape", errors="ignore"))

        for candidate in candidates:
            try:
                payloads.append(json.loads(candidate))
                continue
            except json.JSONDecodeError:
                pass

            # Some app-state scripts wrap JSON inside JavaScript assignments.
            for match in re.finditer(r"[\[{]", candidate):
                try:
                    payload, _ = decoder.raw_decode(candidate[match.start() :])
                except json.JSONDecodeError:
                    continue
                payloads.append(payload)

    return payloads


def first_clean_string(values: list) -> str | None:
    """Return the first non-empty string from a list of candidate values."""
    for value in values:
        if isinstance(value, str):
            cleaned = clean_text(value)
            if cleaned:
                return cleaned
    return None


def extract_listing_dates(payloads: list) -> dict:
    """Extract dateCreated/datePublished/dateModified from parsed payloads."""
    dates = {}
    for field in ("dateCreated", "datePublished", "dateModified"):
        value = first_clean_string([item for payload in payloads for item in find_values_for_key(payload, field)])
        if value:
            dates[field] = value
    return dates


def extract_listing_contact(tree: HTMLParser) -> dict:
    """Extract agent and agency values exposed in listing BI attributes."""
    agency = None
    agent = None

    # imobiliare.ro exposes seller identity in BI attributes on the detail page.
    for node in tree.css("[data-bi-listing-agency], [data-bi-listing-agent]"):
        agency = agency or clean_text(node.attributes.get("data-bi-listing-agency"))
        agent = agent or clean_text(node.attributes.get("data-bi-listing-agent"))

    result = {
        "data_bi_listing_agency": agency,
        "data_bi_listing_agent": agent,
    }

    if agency and agent:
        # If agency and agent match, the listing is treated as owner-posted.
        result["seller_type"] = "owner" if agency == agent else "agency"
    elif agency or agent:
        result["seller_type"] = "unknown"

    return result


def clean_address_value(value: str | None) -> str | None:
    """Normalize address values from JSON-LD or rendered markup."""
    value = clean_text(value)
    if not value:
        return None

    value = html.unescape(value)
    value = re.sub(r"\s*,\s*,+", ",", value)
    value = value.strip(" ,")
    return value or None


def is_portal_company_address(address: dict) -> bool:
    """Filter out the portal's own company address when it appears in JSON-LD."""
    street = clean_address_value(address.get("streetAddress")) or ""
    locality = clean_address_value(address.get("addressLocality")) or ""
    normalized_locality = locality.lower().replace("\u0219", "s")

    return "victor babe" in street.lower() or normalized_locality == "timisoara"


def extract_address_from_json_ld(tree: HTMLParser) -> dict:
    """Read listing address fields from JSON-LD scripts."""
    for node in tree.css('script[type="application/ld+json"]'):
        raw_json = node.text()
        if not raw_json:
            continue

        try:
            data = json.loads(html.unescape(raw_json))
        except json.JSONDecodeError:
            continue

        addresses = [address for address in find_postal_addresses(data) if not is_portal_company_address(address)]
        for address in addresses:
            parsed = {
                "street_address": clean_address_value(address.get("streetAddress")),
                "address_locality": clean_address_value(address.get("addressLocality")),
                "address_region": clean_address_value(address.get("addressRegion")),
                "address_country": clean_address_value(address.get("addressCountry")),
            }
            if any(parsed.values()):
                return parsed

    return {}


def extract_address_from_markup(tree: HTMLParser) -> dict:
    """Read visible address text from the rendered detail page markup."""
    node = tree.css_first('[data-cy="listing-address"]')
    if not node:
        return {}

    text = node.text(separator=" ")
    text = re.sub(r"\s*-\s*Vezi\s+Hart\S*\s*", "", text, flags=re.IGNORECASE)
    text = clean_address_value(text)
    if not text:
        return {}

    parts = [part.strip() for part in text.split(",") if part.strip()]
    result = {"full_address_text": text}

    if len(parts) >= 2:
        result["address_locality"] = parts[0]
        result["address_region"] = parts[-1]
    else:
        result["address_locality"] = text

    return result


def extract_detail_address(tree: HTMLParser) -> dict:
    """Merge structured and rendered address sources."""
    address = extract_address_from_json_ld(tree)
    markup_address = extract_address_from_markup(tree)

    for key, value in markup_address.items():
        if value and not address.get(key):
            address[key] = value

    return address


def extract_detail_metadata(tree: HTMLParser) -> dict:
    """Extract all non-card metadata available on the detail page."""
    payloads = json_payloads_from_scripts(tree)
    metadata = extract_listing_contact(tree)
    metadata.update(extract_listing_dates(payloads))
    return metadata


def resolve_listing_url(client, listing_url: str) -> tuple[str, str | None, dict]:
    """Fetch a detail page, resolve its canonical URL, and extract metadata."""
    try:
        response = fetch_response(client, listing_url)
        tree = HTMLParser(response.text)
        detail_metadata = extract_detail_address(tree)
        detail_metadata.update(extract_detail_metadata(tree))
        canonical = tree.css_first('link[rel="canonical"]')

        if canonical:
            href = canonical.attributes.get("href")
            if href:
                return urljoin(listing_url, href), None, detail_metadata

        return str(response.url), None, detail_metadata
    except RequestException as error:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
        reason = getattr(response, "reason", None)
        if status_code is not None:
            message = f"HTTP {status_code}: {reason or 'request failed'}"
        else:
            message = f"{type(error).__name__}: {error}"
        print(f"Warning: detail URL failed ({message}): {listing_url}")
        return listing_url, message, {}


def chunked_tasks(tasks: list[tuple[int, str]], chunk_count: int) -> list[list[tuple[int, str]]]:
    """Distribute detail-page tasks across worker chunks."""
    chunks = [[] for _ in range(max(1, chunk_count))]
    for position, task in enumerate(tasks):
        chunks[position % len(chunks)].append(task)
    return [chunk for chunk in chunks if chunk]


def resolve_listing_detail_chunk(tasks: list[tuple[int, str]]) -> list[tuple[int, str, str | None, dict]]:
    """Resolve a chunk of detail pages with one reusable HTTP client."""
    results = []

    # Reuse one HTTP client per worker chunk to keep connections warm.
    with create_client() as client:
        for index, listing_url in tasks:
            if DETAIL_REQUEST_DELAY_SECONDS > 0:
                time.sleep(DETAIL_REQUEST_DELAY_SECONDS)

            final_url, detail_error, address = resolve_listing_url(client, listing_url)
            results.append((index, final_url, detail_error, address))

    return results


def resolve_detail_urls(df):
    """Enrich a DataFrame with canonical URLs and detail-page metadata."""
    detail_limit = len(df) if MAX_DETAIL_PAGES is None else min(len(df), MAX_DETAIL_PAGES)
    print(f"Resolving detail URLs: {detail_limit}/{len(df)} with {DETAIL_WORKERS} workers")

    tasks = [(index, df.at[index, "listing_url"]) for index in df.index[:detail_limit]]

    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as executor:
        chunks = chunked_tasks(tasks, DETAIL_WORKERS)
        futures = [executor.submit(resolve_listing_detail_chunk, chunk) for chunk in chunks]
        resolved_count = 0

        for future in as_completed(futures):
            for index, final_url, detail_error, address in future.result():
                resolved_count += 1
                df.at[index, "final_listing_url"] = final_url
                df.at[index, "detail_error"] = detail_error

                for field, value in address.items():
                    df.at[index, field] = value

                print(f"Resolved detail URL {resolved_count}/{detail_limit}: {final_url}")

    return df
