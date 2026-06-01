from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import requests


MOEX_ISS_BASE_URL = "https://iss.moex.com/iss"


@dataclass(frozen=True)
class MoexHistoryRequest:
    secid: str
    from_date: str
    till_date: str
    engine: str = "stock"
    market: str = "index"


class MoexIssClient:
    def __init__(self, base_url: str = MOEX_ISS_BASE_URL, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_index_history(self, request: MoexHistoryRequest) -> pd.DataFrame:
        url = (
            f"{self.base_url}/history/engines/{request.engine}"
            f"/markets/{request.market}/securities/{request.secid}.json"
        )
        params = {
            "from": request.from_date,
            "till": request.till_date,
            "iss.meta": "off",
            "history.columns": ",".join(
                [
                    "BOARDID",
                    "SECID",
                    "TRADEDATE",
                    "SHORTNAME",
                    "NAME",
                    "OPEN",
                    "HIGH",
                    "LOW",
                    "CLOSE",
                    "VALUE",
                    "CAPITALIZATION",
                    "CURRENCYID",
                ]
            ),
        }

        frames: list[pd.DataFrame] = []
        start = 0
        total = None

        while total is None or start < total:
            response = self.session.get(
                url,
                params={**params, "start": start},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()

            history = payload.get("history", {})
            columns = history.get("columns", [])
            rows = history.get("data", [])
            if rows:
                frames.append(pd.DataFrame(rows, columns=columns))

            cursor_rows = payload.get("history.cursor", {}).get("data", [])
            if not cursor_rows:
                break

            _, total, page_size = cursor_rows[0]
            if page_size == 0:
                break
            start += page_size

        if not frames:
            return self._empty_history_frame()

        data = pd.concat(frames, ignore_index=True)
        return normalize_history_frame(data)

    @staticmethod
    def _empty_history_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "boardid",
                "secid",
                "date",
                "shortname",
                "name",
                "open",
                "high",
                "low",
                "close",
                "value",
                "capitalization",
                "currency",
            ]
        )


def normalize_history_frame(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.rename(
        columns={
            "BOARDID": "boardid",
            "SECID": "secid",
            "TRADEDATE": "date",
            "SHORTNAME": "shortname",
            "NAME": "name",
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VALUE": "value",
            "CAPITALIZATION": "capitalization",
            "CURRENCYID": "currency",
        }
    ).copy()

    normalized["date"] = pd.to_datetime(normalized["date"]).dt.date
    numeric_columns = ["open", "high", "low", "close", "value", "capitalization"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    columns = [
        "boardid",
        "secid",
        "date",
        "shortname",
        "name",
        "open",
        "high",
        "low",
        "close",
        "value",
        "capitalization",
        "currency",
    ]
    return normalized[columns].sort_values(["secid", "date"]).reset_index(drop=True)


def combine_histories(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    data = pd.concat(list(frames), ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["secid", "date"]).reset_index(drop=True)

    grouped = data.groupby("secid", group_keys=False)
    previous_close = grouped["close"].shift(1)
    data["return_1d"] = grouped["close"].pct_change()
    data["log_return_1d"] = np.log(data["close"] / previous_close)
    data["range_pct"] = (data["high"] - data["low"]) / data["close"]
    data["gap_pct"] = data["open"] / previous_close - 1

    return data
