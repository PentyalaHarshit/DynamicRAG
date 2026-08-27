"""Open-Meteo weather lookup used by the router's weather agent."""
from typing import Any, Dict

import requests


_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorms",
    96: "thunderstorms with slight hail",
    99: "thunderstorms with heavy hail",
}


def get_weather(location: str) -> Dict[str, Any]:
    """Geocode a location and return its current weather from Open-Meteo."""
    location = location.strip(" .,?!")
    if not location:
        raise ValueError("Please provide a city or location for the weather lookup.")

    geocode_response = requests.get(
        _GEOCODING_URL,
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=8,
    )
    geocode_response.raise_for_status()
    results = geocode_response.json().get("results", [])
    if not results:
        raise ValueError(f"I could not find a location named '{location}'.")

    place = results[0]
    forecast_response = requests.get(
        _FORECAST_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        },
        timeout=8,
    )
    forecast_response.raise_for_status()
    current = forecast_response.json().get("current", {})
    code = current.get("weather_code")

    return {
        "location": ", ".join(
            part for part in [place.get("name"), place.get("admin1"), place.get("country")]
            if part
        ),
        "temperature_c": current.get("temperature_2m"),
        "feels_like_c": current.get("apparent_temperature"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "wind_kmh": current.get("wind_speed_10m"),
        "condition": _WEATHER_CODES.get(code, "unknown conditions"),
        "observed_at": current.get("time"),
    }
