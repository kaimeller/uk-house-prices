import csv
import re
import warnings
import pandas as pd

_TRUE = {"true", "yes", "y"}
_FALSE = {"false", "no", "n"}
_NUM_CLEAN = re.compile(r"^[-+]?[$£€]?\d{1,3}(,\d{3})+(\.\d+)?$|^[-+]?[$£€]\d+(\.\d+)?$")
_LEADING_ZERO = re.compile(r"^[-+]?0\d+")


def _convert_series(s):
    """Infer the best dtype for a string Series. Falls back to the original strings."""
    s = s.replace("", pd.NA)
    non_null = s.dropna()
    if non_null.empty:
        return s.astype("string")

    # Boolean (only if every value is clearly boolean)
    lowered = non_null.str.lower()
    if lowered.isin(_TRUE | _FALSE).all():
        return s.str.lower().map(lambda v: pd.NA if pd.isna(v) else v in _TRUE).astype("boolean")

    # Values with leading zeros (e.g. "007") are identifiers: keep as text
    if non_null.str.match(_LEADING_ZERO).any():
        return s.astype("string")

    # Numeric
    if True:
        candidate = s
        # Strip thousands separators / currency symbols, but only if they follow a strict pattern
        if non_null.str.match(_NUM_CLEAN).any():
            candidate = s.str.replace(r"[,$£€]", "", regex=True)
        try:
            nums = pd.to_numeric(candidate)
            if (nums.dropna() % 1 == 0).all():
                return nums.astype("Int64")
            return nums.astype("float64")
        except (ValueError, TypeError):
            pass

    # Datetime
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(s)
    except (ValueError, TypeError):
        pass

    return s.astype("string")


def read_nomis_csv(path, columns, name_offset=(-1, 0), convert_types=True,
                         dtypes=None, encoding="utf-8-sig"):
    """
    Read a CSV containing several tables separated by blank space.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    columns : list[str]
        The column names every table starts with (the header row, in order).
    name_offset : (int, int)
        (row_offset, col_offset) from the header's first cell (top-left) to the
        cell holding the table name. Default (-1, 0) = the cell directly above
        the first header cell. Use (0, -1) for the cell to the left, etc.
    convert_types : bool
        If True, infer and convert each column to bool, int (nullable Int64),
        float, datetime, or string. Empty cells become NA.
    dtypes : dict[str, str | type] | None
        Optional per-column overrides, e.g. {"ID": "string", "Date": "datetime64[ns]"}.
        Applied after inference and takes priority over it.
    encoding : str
        File encoding.

    Returns
    -------
    dict[str, pd.DataFrame]
        Table name -> DataFrame. Duplicate names get a numeric suffix.
    """
    with open(path, newline="", encoding=encoding) as f:
        rows = [[c.strip() for c in row] for row in csv.reader(f)]

    n = len(columns)
    d_row, d_col = name_offset

    def cell(r, c):
        if 0 <= r < len(rows) and 0 <= c < len(rows[r]):
            return rows[r][c]
        return ""

    tables = {}
    for r, row in enumerate(rows):
        # Find every position in this row where the header sequence starts
        # (handles tables placed at any column offset, even side by side).
        for c in range(len(row) - n + 1):
            if row[c:c + n] != list(columns):
                continue

            name = cell(r + d_row, c + d_col) or f"table_{len(tables) + 1}"

            # Collect data rows until the table's column span is entirely blank
            data = []
            rr = r + 1
            while rr < len(rows):
                values = [cell(rr, c + i) for i in range(n)]
                if not any(values):
                    break
                data.append(values)
                rr += 1

            # Make the name unique if it repeats
            base, k = name, 2
            while name in tables:
                name, k = f"{base}_{k}", k + 1

            df = pd.DataFrame(data, columns=columns)

            if convert_types:
                for col in df.columns:
                    df[col] = _convert_series(df[col])
            if dtypes:
                for col, dt in dtypes.items():
                    if col in df.columns:
                        df[col] = df[col].replace("", pd.NA).astype(dt) if df[col].dtype == object else df[col].astype(dt)

            tables[name] = df

    return tables
