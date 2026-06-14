import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import httpx
from selectolax.parser import HTMLParser

from scraper.core.config import DETAIL_REQUEST_DELAY_SECONDS, DETAIL_WORKERS, MAX_DETAIL_PAGES
from scraper.core.http_client import fetch_response
from scraper.sites.imobiliare.parser import clean_text


def find_postal_addresses(value) -> list[dict]:
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


def clean_address_value(value: str | None) -> str | None:
    value = clean_text(value)
    if not value:
        return None

    value = html.unescape(value)
    value = re.sub(r"\s*,\s*,+", ",", value)
    value = value.strip(" ,")
    return value or None


def is_portal_company_address(address: dict) -> bool:
    street = clean_address_value(address.get("streetAddress")) or ""
    locality = clean_address_value(address.get("addressLocality")) or ""
    normalized_locality = locality.lower().replace("\u0219", "s")

    return "victor babe" in street.lower() or normalized_locality == "timisoara"


def extract_address_from_json_ld(tree: HTMLParser) -> dict:
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
    address = extract_address_from_json_ld(tree)
    markup_address = extract_address_from_markup(tree)

    for key, value in markup_address.items():
        if value and not address.get(key):
            address[key] = value

    return address


def resolve_listing_url(client: httpx.Client, listing_url: str) -> tuple[str, str | None, dict]:
    try:
        response = fetch_response(client, listing_url)
        tree = HTMLParser(response.text)
        address = extract_detail_address(tree)
        canonical = tree.css_first('link[rel="canonical"]')

        if canonical:
            href = canonical.attributes.get("href")
            if href:
                return urljoin(listing_url, href), None, address

        return str(response.url), None, address
    except httpx.HTTPStatusError as error:
        status_code = error.response.status_code
        message = f"HTTP {status_code}: {error.response.reason_phrase}"
        print(f"Warning: detail URL failed ({message}): {listing_url}")
        return listing_url, message, {}
    except httpx.HTTPError as error:
        message = f"{type(error).__name__}: {error}"
        print(f"Warning: detail URL failed ({message}): {listing_url}")
        return listing_url, message, {}


def resolve_listing_detail(index_and_url: tuple[int, str]) -> tuple[int, str, str | None, dict]:
    index, listing_url = index_and_url

    if DETAIL_REQUEST_DELAY_SECONDS > 0:
        time.sleep(DETAIL_REQUEST_DELAY_SECONDS)

    with httpx.Client(follow_redirects=True) as client:
        final_url, detail_error, address = resolve_listing_url(client, listing_url)

    return index, final_url, detail_error, address


def resolve_detail_urls(df):
    detail_limit = len(df) if MAX_DETAIL_PAGES is None else min(len(df), MAX_DETAIL_PAGES)
    print(f"Resolving detail URLs: {detail_limit}/{len(df)} with {DETAIL_WORKERS} workers")

    tasks = [(index, df.at[index, "listing_url"]) for index in df.index[:detail_limit]]

    with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as executor:
        futures = [executor.submit(resolve_listing_detail, task) for task in tasks]

        for position, future in enumerate(as_completed(futures), start=1):
            index, final_url, detail_error, address = future.result()
            df.at[index, "final_listing_url"] = final_url
            df.at[index, "detail_error"] = detail_error

            for field, value in address.items():
                df.at[index, field] = value

            print(f"Resolved detail URL {position}/{detail_limit}: {final_url}")

    return df
