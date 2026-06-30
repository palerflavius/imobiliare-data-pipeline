import json
import sys
from pathlib import Path


BASE_URL = "https://www.imobiliare.ro"


def offer_property_path(offer_type: str, property_type: str) -> str:
    """Map normalized scraper dimensions to the URL path used by imobiliare.ro."""
    paths = {
        ("sale", "apartments"): "vanzare-apartamente",
        ("sale", "houses-villas"): "vanzare-case-vile",
        ("sale", "lands"): "vanzare-terenuri",
        ("rent", "apartments"): "inchirieri-apartamente",
        ("rent", "houses-villas"): "inchirieri-case-vile",
    }
    try:
        return paths[(offer_type, property_type)]
    except KeyError as error:
        raise ValueError(f"Unsupported search: offer_type={offer_type}, property_type={property_type}") from error


def start_url(
    site_name: str,
    county_slug: str,
    city_slug: str,
    area_slug: str | None,
    offer_type: str,
    property_type: str,
) -> str:
    """Build the search URL for one GitHub Actions matrix target."""
    if site_name != "imobiliare.ro":
        raise ValueError(f"Unsupported site_name: {site_name}")

    path = offer_property_path(offer_type, property_type)

    # Bucharest URLs use the city path directly, not the judetul-* county form.
    if county_slug == "bucuresti":
        url = f"{BASE_URL}/{path}/bucuresti"
        if area_slug:
            url = f"{url}/{area_slug}"
        return url

    # city=all means scrape the county page once; rows can still be partitioned later by detected city.
    if city_slug == "all":
        return f"{BASE_URL}/{path}/judetul-{county_slug}"

    url = f"{BASE_URL}/{path}/judetul-{county_slug}/{city_slug}"
    if area_slug:
        url = f"{url}/{area_slug}"
    return url


def add_target(targets: list[dict], defaults: dict, county_slug: str, city_slug: str, area_slug: str | None = None) -> None:
    """Append one geographic scraper target to the GitHub Actions matrix payload."""
    site_name = defaults["site_name"]
    searches = defaults.get("searches")
    if not searches:
        searches = [{"offer_type": defaults["offer_type"], "property_type": defaults["property_type"]}]

    searches_by_offer = {}
    for search in searches:
        searches_by_offer.setdefault(search["offer_type"], []).append(
            {
                "offer_type": search["offer_type"],
                "property_type": search["property_type"],
                "start_url": start_url(
                    site_name,
                    county_slug,
                    city_slug,
                    area_slug,
                    search["offer_type"],
                    search["property_type"],
                ),
            }
        )

    targets.append(
        {
            "site_name": site_name,
            "county_slug": county_slug,
            "city_slug": city_slug,
            "area_slug": area_slug or "",
            "sale_searches": searches_by_offer.get("sale", []),
            "rent_searches": searches_by_offer.get("rent", []),
        }
    )


def build_targets(config: dict) -> list[dict]:
    """Expand the JSON target config into a flat list of scraper jobs."""
    defaults = config["defaults"]
    targets = []

    for county in config["counties"]:
        county_slug = county["county_slug"]
        mode = county.get("mode", "municipality")
        municipality_slug = county.get("municipality_slug")

        # Modes define the county-level target; optional localities add extra targets below.
        if mode == "all":
            add_target(targets, defaults, county_slug, "all")
        elif mode == "municipality":
            if not municipality_slug:
                raise ValueError(f"County {county_slug} needs municipality_slug for mode=municipality")
            add_target(targets, defaults, county_slug, municipality_slug)
        elif mode == "areas":
            if not municipality_slug:
                raise ValueError(f"County {county_slug} needs municipality_slug for mode=areas")
            for area_slug in county.get("areas", []):
                add_target(targets, defaults, county_slug, municipality_slug, area_slug)
        else:
            raise ValueError(f"Unsupported mode for {county_slug}: {mode}")

        for locality in county.get("localities", []):
            # A locality can be a simple city slug or an object with area-level targets.
            if isinstance(locality, str):
                add_target(targets, defaults, county_slug, locality)
                continue

            city_slug = locality["city_slug"]
            area_slugs = locality.get("areas") or [""]
            for area_slug in area_slugs:
                add_target(targets, defaults, county_slug, city_slug, area_slug or None)

    return targets


def main() -> None:
    """Print a compact JSON matrix for GitHub Actions."""
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else "scraper/targets/imobiliare.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    targets = build_targets(config)

    if not targets:
        raise RuntimeError(f"No targets generated from {config_path}")

    print(json.dumps({"target": targets}, separators=(",", ":")))


if __name__ == "__main__":
    main()
