#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import io
import os
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd


@dataclass
class Crash:
    crash_no: int
    peak_date: pd.Timestamp
    peak_value: float
    start_idx: int
    start_date: pd.Timestamp
    start_value: float
    end_idx: Optional[int] = None
    end_date: Optional[pd.Timestamp] = None
    end_value: Optional[float] = None


def load_csv(source: str) -> pd.DataFrame:
    # source can be a file path or a URL (Internetadresse)
    if source.startswith("http://") or source.startswith("https://"):
        # pandas can read URLs directly
        df = pd.read_csv(source)
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Datei nicht gefunden: {source}")
        df = pd.read_csv(source)
    return df


def detect_crashes(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    crash_threshold: float = 0.20,
    eps: float = 1e-12,
) -> List[Crash]:
    # Crash definition:
    # - begins when drawdown from previous all-time high (Allzeithoch - ATH) is <= -crash_threshold
    # - counted once no matter how long it lasts
    # - ends when a new ATH is reached (value > previous ATH)
    df = df.sort_values(date_col).reset_index(drop=True)

    peak_value = float("-inf")
    peak_date = None

    in_crash = False
    crashes: List[Crash] = []
    crash_no = 0
    active_crash: Optional[Crash] = None

    for i, row in df.iterrows():
        d = row[date_col]
        v = float(row[value_col])

        # Update ATH if we are NOT currently in crash
        if not in_crash and v > peak_value + eps:
            peak_value = v
            peak_date = d

        if not in_crash:
            # Check crash start relative to last ATH
            if peak_value > 0:
                drawdown = v / peak_value - 1.0
                if drawdown <= -crash_threshold:
                    crash_no += 1
                    in_crash = True
                    active_crash = Crash(
                        crash_no=crash_no,
                        peak_date=peak_date,
                        peak_value=peak_value,
                        start_idx=i,
                        start_date=d,
                        start_value=v,
                    )
                    crashes.append(active_crash)
        else:
            # Crash ends only when a NEW ATH is reached (strictly greater than the pre-crash ATH)
            if v > active_crash.peak_value + eps:
                active_crash.end_idx = i
                active_crash.end_date = d
                active_crash.end_value = v

                # Exit crash and update ATH
                in_crash = False
                peak_value = v
                peak_date = d
                active_crash = None

    return crashes


def forward_return(df: pd.DataFrame, value_col: str, start_idx: int, months_forward: int) -> Optional[float]:
    j = start_idx + months_forward
    if j >= len(df):
        return None
    start_v = float(df.loc[start_idx, value_col])
    future_v = float(df.loc[j, value_col])
    if start_v == 0:
        return None
    return future_v / start_v - 1.0


def main():
    ap = argparse.ArgumentParser(
        description="Crashes (>=20%% Drawdown vom Allzeithoch) zählen und 12M-Rendite nach Crash-Beginn ausgeben."
    )
    ap.add_argument(
        "--source",
        default="https://github.com/VaraNiN/AverageReturn/blob/main/data/msci_world_curvo.csv?plain=1",
        help="Pfad zur Datei oder URL (Internetadresse) zur CSV-Datei (Comma-Separated Values).",
    )
    ap.add_argument("--crash", type=float, default=0.20, help="Crash-Schwelle als Dezimalzahl, z.B. 0.20 für 20%%.")
    ap.add_argument("--months", type=int, default=12, help="Anzahl Monate nach Crash-Beginn für die Folgerendite.")
    args = ap.parse_args()

    df = load_csv(args.source)

    # Be robust: assume first col is Date, second col is the index level
    if "Date" not in df.columns:
        raise ValueError(f"Spalte 'Date' nicht gefunden. Gefundene Spalten: {list(df.columns)}")

    value_col = None
    for c in df.columns:
        if c != "Date":
            value_col = c
            break
    if value_col is None:
        raise ValueError("Keine Wertespalte gefunden (erwartet: neben 'Date' mindestens eine zweite Spalte).")

    # Parse dates like 12/1978
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%Y", errors="raise")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["Date", value_col]).sort_values("Date").reset_index(drop=True)

    crashes = detect_crashes(df, date_col="Date", value_col=value_col, crash_threshold=args.crash)

    print(f"Anzahl Crashes (>= {args.crash*100:.0f}% vom vorherigen Allzeithoch): {len(crashes)}")

    rows = []
    for c in crashes:
        r = forward_return(df, value_col=value_col, start_idx=c.start_idx, months_forward=args.months)
        rows.append(
            {
                "Crash": c.crash_no,
                "PeakDate": c.peak_date.strftime("%Y-%m"),
                "StartDate": c.start_date.strftime("%Y-%m"),
                "EndDate": c.end_date.strftime("%Y-%m") if c.end_date is not None else None,
                f"Return_{args.months}M": r,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        # Stats ignoring missing forward returns at dataset end
        valid = out[f"Return_{args.months}M"].dropna()
        mean_r = float(valid.mean()) if len(valid) else None
        median_r = float(valid.median()) if len(valid) else None

        # Pretty print
        out_fmt = out.copy()
        out_fmt[f"Return_{args.months}M"] = out_fmt[f"Return_{args.months}M"].map(
            lambda x: (f"{x*100:.2f}%" if pd.notna(x) else None)
        )
        print("\nCrashes und Rendite nach Crash-Beginn:")
        print(out_fmt.to_string(index=False))

        print("\nZusammenfassung der Folgerenditen:")
        print(f"- Durchschnitt: {mean_r*100:.2f}%" if mean_r is not None else "- Durchschnitt: n/a")
        print(f"- Median:       {median_r*100:.2f}%" if median_r is not None else "- Median: n/a")
    else:
        print("Keine Crashes gefunden.")


if __name__ == "__main__":
    main()
