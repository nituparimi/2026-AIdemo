from __future__ import annotations
from typing import Any, Optional
import httpx  # type: ignore[reportMissingImports]
from mcp.server.fastmcp import FastMCP

# ---------- Init ----------
mcp = FastMCP("weather")

# ---------- Constants ----------
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0 (contact: you@example.com)"  # add contact per NWS guidance
MAX_CHARS = 3000  # hard cap to keep LLM payloads small
MAX_ITEMS = 20    # safety bound on count

# ---------- HTTP ----------
async def make_nws_request(url: str) -> dict[str, Any] | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(url, headers=headers, timeout=30.0)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

# ---------- Helpers ----------
def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)

def _to_int(v: Any, default: int) -> int:
    try:
        return max(1, min(int(v), MAX_ITEMS))
    except Exception:
        return default

def _brief_alert(feature: dict, include_expires: bool = True) -> str:
    p = feature.get("properties", {})
    ev = p.get("event", "?")
    area = p.get("areaDesc", "?")
    if len(area) > 120:
        area = area[:117] + "…"
    ends = p.get("ends") or p.get("expires") or "N/A"
    tail = f" (until {ends})" if include_expires else ""
    return f"• {ev} — {area}{tail}"

# Optional verbose formatter (kept for debugging)
def format_alert(feature: dict) -> str:
    p = feature.get("properties", {})
    return (
        f"Event: {p.get('event','Unknown')}\n"
        f"Area: {p.get('areaDesc','Unknown')}\n"
        f"Severity: {p.get('severity','Unknown')}\n"
        f"Description: {p.get('description','No description available')}\n"
        f"Instructions: {p.get('instruction','No specific instructions provided')}"
    )

# ---------- Tools ----------
@mcp.tool()
async def get_alerts(
    state: str,
    event_filter: Optional[str] = None,
    limit: int | str = 5,
    include_expires: bool | str = True,
) -> str:
    """
    Return a compact list of alerts for a US state.
    Args (strings also accepted): state="CA", event_filter="heat", limit=5, include_expires=true
    """
    state = (state or "").upper().strip()
    data = await make_nws_request(f"{NWS_API_BASE}/alerts/active/area/{state}")

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."
    feats = data.get("features") or []
    if not feats:
        return "No active alerts for this state."

    # filter
    if event_filter:
        q = str(event_filter).lower()
        feats = [f for f in feats if q in (f.get("properties", {}).get("event", "") or "").lower()]
        if not feats:
            return f"No matching alerts for filter '{event_filter}'."

    # coerce + cap
    n = _to_int(limit, 5)
    inc = _to_bool(include_expires)

    # brief + join
    lines = [_brief_alert(f, inc) for f in feats[:n]]
    out = "\n".join(lines) or "No matching alerts."

    # final guard
    return out[:MAX_CHARS]

@mcp.tool()
async def ping() -> str:
    """Health check."""
    return "🏓 Pong! Weather service is online and ready to provide weather alerts."

@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    return f"Resource echo: {message}"

# ---------- Run ----------
if __name__ == "__main__":
    try:
        mcp.run()
    except KeyboardInterrupt:
        pass

# ---------- Forecast Helper ----------

async def get_location_data(latitude: float, longitude: float):
    """Retrieve NWS grid and forecast URLs for a location."""

    if not (-90 <= latitude <= 90):
        raise ValueError("Latitude must be between -90 and 90.")

    if not (-180 <= longitude <= 180):
        raise ValueError("Longitude must be between -180 and 180.")

    url = f"{NWS_API_BASE}/points/{latitude},{longitude}"

    data = await make_nws_request(url)

    if not data:
        return None

    return data.get("properties", {})


# ---------- Tool 3: Weather Forecast ----------

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """
    Get the weather forecast for a location using latitude and longitude.
    Example: latitude=37.3382, longitude=-121.8863
    """

    location = await get_location_data(latitude, longitude)

    if not location:
        return "Unable to retrieve location forecast information."

    forecast_url = location.get("forecast")

    if not forecast_url:
        return "Forecast URL unavailable for this location."

    data = await make_nws_request(forecast_url)

    if not data:
        return "Unable to retrieve forecast."

    periods = data.get("properties", {}).get("periods", [])

    if not periods:
        return "No forecast periods available."

    results = []

    for period in periods[:5]:
        results.append(
            f"{period.get('name')}: "
            f"{period.get('temperature')}°"
            f"{period.get('temperatureUnit')}, "
            f"{period.get('shortForecast')}, "
            f"Wind: {period.get('windSpeed')} "
            f"{period.get('windDirection')}"
        )

    return "\n".join(results)


# ---------- Tool 4: Hourly Forecast ----------

@mcp.tool()
async def get_hourly_forecast(
    latitude: float,
    longitude: float,
    limit: int = 12
) -> str:
    """
    Get hourly weather forecast for a location.
    Limit is between 1 and 24.
    """

    location = await get_location_data(latitude, longitude)

    if not location:
        return "Unable to retrieve location information."

    hourly_url = location.get("forecastHourly")

    if not hourly_url:
        return "Hourly forecast unavailable."

    data = await make_nws_request(hourly_url)

    if not data:
        return "Unable to retrieve hourly forecast."

    periods = data.get("properties", {}).get("periods", [])

    limit = max(1, min(limit, 24))

    results = []

    for period in periods[:limit]:
        results.append(
            f"{period.get('startTime')}: "
            f"{period.get('temperature')}°"
            f"{period.get('temperatureUnit')}, "
            f"{period.get('shortForecast')}, "
            f"Wind: {period.get('windSpeed')}"
        )

    return "\n".join(results) or "No hourly forecast available."


# ---------- Tool 5: Current Conditions ----------

@mcp.tool()
async def get_current_conditions(
    latitude: float,
    longitude: float
) -> str:
    """
    Get the latest available weather observation for a location.
    """

    location = await get_location_data(latitude, longitude)

    if not location:
        return "Unable to retrieve location information."

    stations_url = location.get("observationStations")

    if not stations_url:
        return "Observation stations unavailable."

    stations_data = await make_nws_request(stations_url)

    if not stations_data:
        return "Unable to retrieve observation stations."

    stations = stations_data.get("features", [])

    if not stations:
        return "No observation stations found."

    station_id = stations[0].get("properties", {}).get("stationIdentifier")

    if not station_id:
        return "Station identifier unavailable."

    observation_url = (
        f"{NWS_API_BASE}/stations/{station_id}/observations/latest"
    )

    data = await make_nws_request(observation_url)

    if not data:
        return "Unable to retrieve current conditions."

    p = data.get("properties", {})

    temperature = p.get("temperature", {}).get("value")
    humidity = p.get("relativeHumidity", {}).get("value")
    wind = p.get("windSpeed", {}).get("value")
    description = p.get("textDescription", "Unknown")

    return (
        f"Station: {station_id}\n"
        f"Conditions: {description}\n"
        f"Temperature: {temperature} °C\n"
        f"Humidity: {humidity}%\n"
        f"Wind Speed: {wind} m/s"
    )


# ---------- Tool 6: Detailed Alert ----------

@mcp.tool()
async def get_active_alert_details(state: str) -> str:
    """
    Get detailed information about active weather alerts for a US state.
    """

    state = state.upper().strip()

    if len(state) != 2 or not state.isalpha():
        return "Please provide a valid two-letter state code."

    data = await make_nws_request(
        f"{NWS_API_BASE}/alerts/active/area/{state}"
    )

    if not data:
        return "Unable to retrieve alert details."

    features = data.get("features", [])

    if not features:
        return "No active alerts found."

    results = []

    for feature in features[:5]:
        results.append(format_alert(feature))

    return "\n\n---\n\n".join(results)[:MAX_CHARS]
