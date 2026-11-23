"""Reading data files and pasted text into the wire format."""
from __future__ import annotations

import csv
import io
import os
import tempfile

import pandas as pd

from .core.dataset import frame_to_wire
from .core.errors import AnalysisError

MAX_ROWS = 500_000
EXTENSIONS = [".csv", ".tsv", ".txt", ".tab", ".xlsx", ".xlsm", ".xls", ".json", ".parquet",
              ".sav", ".zsav", ".dta", ".sas7bdat", ".xpt", ".rds", ".rdata", ".rda"]


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    names, seen = [], {}
    for i, c in enumerate(df.columns):
        n = str(c).strip()
        if not n or n.lower().startswith("unnamed:"):
            n = f"Column {i + 1}"
        if n in seen:
            seen[n] += 1
            n = f"{n} ({seen[n]})"
        else:
            seen[n] = 1
        names.append(n)
    df.columns = names
    if len(df) > MAX_ROWS:
        raise AnalysisError(f"This file has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    return df


def _sniff(text: str) -> str:
    sample = "\n".join(text.splitlines()[:30])
    if "\t" in sample:
        return "\t"
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;|\t").delimiter
    except csv.Error:
        return ","


def _has_header(rows: list[list[str]]) -> bool:
    """Header if the first row has a text cell above an otherwise numeric column."""
    if len(rows) < 2:
        return False

    def num(s):
        try:
            float(s.replace(",", ""))
            return True
        except ValueError:
            return False
    first, rest = rows[0], rows[1:]
    if all(num(c) for c in first if c.strip()):
        return False
    return True


def read_text(text: str, name: str = "Pasted data") -> list[dict]:
    text = text.lstrip("﻿")
    if not text.strip():
        raise AnalysisError("There is nothing to import.")
    sep = _sniff(text)
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))
    rows = [r for r in rows if any(c.strip() for c in r)]
    header = 0 if _has_header(rows) else None
    df = pd.read_csv(io.StringIO(text), sep=sep, header=header, dtype=object,
                     skip_blank_lines=True, keep_default_na=True, engine="python")
    if header is None:
        df.columns = [f"Column {i + 1}" for i in range(df.shape[1])]
    return [{"name": name, **frame_to_wire(_tidy(df), infer=True)}]


def read_file(filename: str, content: bytes) -> list[dict]:
    """Returns a list of sheets: ``[{"name", "columns": [...]}]``."""
    base = os.path.basename(filename)
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext in (".csv", ".tsv", ".txt", ".tab", ""):
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return read_text(content.decode(enc), stem)
            except UnicodeDecodeError:
                continue
        raise AnalysisError("The file encoding could not be detected.")
    if ext in (".xlsx", ".xlsm", ".xls"):
        try:
            book = pd.read_excel(io.BytesIO(content), sheet_name=None, dtype=object)
        except ImportError:
            raise AnalysisError("Reading old .xls files needs the 'xlrd' package; save the file as .xlsx instead.") from None
        out = []
        for sheet, df in book.items():
            df = _tidy(df)
            if df.shape[1]:
                out.append({"name": str(sheet) if len(book) > 1 else stem, **frame_to_wire(df, infer=True)})
        if not out:
            raise AnalysisError("The workbook has no data.")
        return out
    if ext == ".json":
        return [{"name": stem, **frame_to_wire(_tidy(pd.read_json(io.BytesIO(content))), infer=True)}]
    if ext == ".parquet":
        return [{"name": stem, **frame_to_wire(_tidy(pd.read_parquet(io.BytesIO(content))), infer=True)}]
    if ext in (".sav", ".zsav", ".dta", ".sas7bdat", ".xpt", ".rds", ".rdata", ".rda"):
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as fh:
            fh.write(content)
            path = fh.name
        try:
            if ext in (".rds", ".rdata", ".rda"):
                from .core.rbridge import pool
                return [{"name": stem, **pool().call({"kind": "read", "path": path})}]
            if ext in (".sav", ".zsav"):
                df = pd.read_spss(path)
            elif ext == ".dta":
                df = pd.read_stata(path)
            else:
                df = pd.read_sas(path, encoding="latin-1")
            return [{"name": stem, **frame_to_wire(_tidy(df), infer=True)}]
        finally:
            os.unlink(path)
    raise AnalysisError(f"“{ext}” files are not supported. Supported: {', '.join(EXTENSIONS)}")
