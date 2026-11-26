import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_from_string, column_index_from_string


def read_xlsx_range(file, sheet_name, corner_1, corner_2):
    """
    Read a rectangular block of an Excel sheet into a single DataFrame.

    The first row of the block is used as the column names.

    Parameters
    ----------
    file : str | path-like | file-like
        Path to the .xlsx file.
    sheet_name : str
        Name of the worksheet.
    corner_1, corner_2 : str
        Cell references of two opposite corners of the block, e.g. "B3" and
        "F20". They can be given in any order (top-left/bottom-right or
        top-right/bottom-left, etc.). The block must include the header row.

    Returns
    -------
    pd.DataFrame
    """
    # Work out the bounds regardless of which corners were supplied
    def parse(ref):
        col, row = coordinate_from_string(ref.replace("$", "").upper())
        return row, column_index_from_string(col)

    (r1, c1), (r2, c2) = parse(corner_1), parse(corner_2)
    min_row, max_row = sorted((r1, r2))
    min_col, max_col = sorted((c1, c2))

    wb = load_workbook(file, read_only=True)
    try:
        if sheet_name not in wb.sheetnames:
            raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {wb.sheetnames}")
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(min_row=min_row, max_row=max_row,
                                 min_col=min_col, max_col=max_col, values_only=True))
    finally:
        wb.close()

    if not rows:
        raise ValueError("The selected range is empty.")

    # Pad in case the sheet's stored dimensions end before the requested range
    width = max_col - min_col + 1
    rows = [tuple(r) + (None,) * (width - len(r)) for r in rows]

    header, *data = rows
    columns = []
    for i, h in enumerate(header):
        name = str(h).strip() if h is not None and str(h).strip() else f"Unnamed_{i + 1}"
        columns.append(name)

    return pd.DataFrame(data, columns=columns).infer_objects()
