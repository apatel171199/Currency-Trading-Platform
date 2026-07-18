from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import requests


@dataclass(frozen=True)
class MarketRequest: #Describes the market data requested from Twelve Data

    symbol: str
    interval: str = "15min"
    output_size: int = 500
    timezone: str = "UTC"


class MarketDataProvider: #Downloads and validates market data from Twelve Data

    BASE_URL = "https://api.twelvedata.com/time_series"
    REQUIRED_COLUMNS = {"Open", "High", "Low", "Close"}

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TWELVE_DATA_API_KEY")

        if not self.api_key:
            raise RuntimeError("Twelve Data API key was not found. "
                "Set the TWELVE_DATA_API_KEY environment variable.")

    def download(self, request: MarketRequest) -> pd.DataFrame:
        self._validate_request(request)

        parameters = {
            "symbol": request.symbol,
            "interval": request.interval,
            "outputsize": request.output_size,
            "timezone": request.timezone,
            "order": "ASC",
            "format": "JSON",
            "apikey": self.api_key,}

        try:
            response = requests.get(
                self.BASE_URL,
                params=parameters,
                timeout=30,)
            response.raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(f"Could not connect to Twelve Data: {error}") from error

        try:
            payload = response.json()
        except requests.JSONDecodeError as error:
            raise RuntimeError("Twelve Data returned an invalid JSON response.") from error

        if payload.get("status") == "error":
            message = payload.get(
                "message",
                "Twelve Data returned an unknown API error.",)
            raise RuntimeError(message)

        values = payload.get("values")

        if not values:
            raise RuntimeError(f"No market data was returned for {request.symbol}.")

        data = pd.DataFrame(values)

        data = data.rename(
            columns={
                "datetime": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",})

        missing_columns = self.REQUIRED_COLUMNS.difference(
            data.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise RuntimeError(f"Downloaded data is missing columns: {missing}")

        numeric_columns = [
            "Open",
            "High",
            "Low",
            "Close",]

        for column in numeric_columns:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",)

        data["Datetime"] = pd.to_datetime(
            data["Datetime"],
            errors="coerce",)

        cleaned_data = (
            data.dropna(
                subset=[
                    "Datetime",
                    "Open",
                    "High",
                    "Low",
                    "Close",])
            .drop_duplicates(subset="Datetime")
            .sort_values("Datetime")
            .set_index("Datetime"))

        if cleaned_data.empty:
            raise RuntimeError("The API response contained no usable price rows.")

        return cleaned_data

    def _validate_request(
        self,
        request: MarketRequest, ) -> None:
        if not request.symbol.strip():
            raise ValueError("Market symbol cannot be empty.")

        if request.output_size <= 0:
            raise ValueError("Output size must be greater than zero.")