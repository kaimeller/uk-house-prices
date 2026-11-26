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


def read_multi_table_csv(path, first_column, name_offset=(-1, 0), header_gap=1,
                         convert_types=True, dtypes=None, encoding="utf-8-sig"):
    """
    Read a CSV containing several tables separated by blank space.

    Only the left-most column name is needed to locate tables. The rest of the
    column names are read from the file: the header runs from that cell to the
    right until the first blank cell (or the next occurrence of `first_column`,
    for tables placed directly side by side).

    Parameters
    ----------
    path : str
        Path to the CSV file.
    first_column : str
        Name of the left-most column; marks the start of every table.
    name_offset : (int, int)
        (row_offset, col_offset) from the header's first cell (top-left) to the
        cell holding the table name. Default (-1, 0) = the cell directly above
        the first header cell. Use (0, -1) for the cell to the left, etc.
    header_gap : int
        Number of rows between the header row and the first data row.
    convert_types : bool
        If True, infer and convert each column to bool, int (nullable Int64),
        float, datetime, or string. Empty cells become NA.
    dtypes : dict[str, str | type] | None
        Optional per-column overrides, e.g. {"ID": "string"}. Applied after
        inference and takes priority over it.
    encoding : str
        File encoding.

    Returns
    -------
    dict[str, pd.DataFrame]
        Table name -> DataFrame. Duplicate names get a numeric suffix.
    """
    with open(path, newline="", encoding=encoding) as f:
        rows = [[c.strip() for c in row] for row in csv.reader(f)]

    d_row, d_col = name_offset
    first_column = first_column.strip()

    def cell(r, c):
        if 0 <= r < len(rows) and 0 <= c < len(rows[r]):
            return rows[r][c]
        return ""

    tables = {}
    consumed = set()  # (row, col) cells already part of a parsed table

    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if value != first_column or (r, c) in consumed:
                continue

            # Read the header rightwards until a blank cell or another table's start
            columns = [first_column]
            cc = c + 1
            while cell(r, cc) and cell(r, cc) != first_column:
                columns.append(cell(r, cc))
                cc += 1
            n = len(columns)

            name = cell(r + d_row, c + d_col) or f"table_{len(tables) + 1}"

            # Collect data rows until the table's column span is entirely blank
            data = []
            rr = r + 1 + header_gap
            while rr < len(rows):
                values = [cell(rr, c + i) for i in range(n)]
                if not any(values):
                    break
                data.append(values)
                rr += 1

            # Don't re-detect cells inside this table as new table starts
            consumed.update((i, c + j) for i in range(r, rr) for j in range(n))

            # Make the name unique if it repeats
            base, k = name, 2
            while name in tables:
                name, k = f"{base}_{k}", k + 1

            df = pd.DataFrame(data, columns=columns)

            if convert_types:
                # Work by position so duplicate column names don't break anything
                converted = [_convert_series(df.iloc[:, i]) for i in range(n)]
                df = pd.concat(converted, axis=1)
                df.columns = columns
            if dtypes:
                for col, dt in dtypes.items():
                    if col in df.columns:
                        s = df[col]
                        if s.dtype == object:
                            s = s.replace("", pd.NA)
                        df[col] = s.astype(dt)

            tables[name] = df

    return tables
