from __future__ import annotations

from pathlib import Path

from config import EXCEL_PATH
from tools.db import get_connection
from tools.importer import import_workbook


def import_excel(path: str | None = None) -> dict:
    excel_path = Path(path or EXCEL_PATH)
    if not excel_path or not excel_path.exists():
        raise FileNotFoundError("Excel path is not configured or does not exist.")
    conn = get_connection()
    result = import_workbook(excel_path, conn)
    conn.close()
    return result


def sample_items(limit: int) -> None:
    conn = get_connection()
    cursor = conn.execute(
        """
        SELECT name, w_mm, d_mm, h_mm, price_unit_ex_vat, has_led, source_sheet, source_row
        FROM items
        WHERE price_unit_ex_vat IS NOT NULL AND price_unit_ex_vat > 0
          AND (w_mm IS NOT NULL OR d_mm IS NOT NULL OR h_mm IS NOT NULL)
        ORDER BY id
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    print(f"sample rows: {len(rows)}")
    for item in rows:
        dims = "x".join(str(val) for val in (item["w_mm"], item["d_mm"], item["h_mm"]) if val)
        light_flag = "yes" if item["has_led"] else "no"
        price = item["price_unit_ex_vat"]
        print(
            f"name={item['name'] or '-'} | dims={dims or '-'} | price={price:.2f} | light={light_flag} | "
            f"sheet={item['source_sheet']} row={item['source_row']}"
        )


def main(path: str | None = None) -> None:
    result = import_excel(path)
    print(
        "Imported {inserted} rows into DB, detected sheets {detected_sheets}, skipped {skipped_sheets}.".format(
            inserted=result["inserted"],
            detected_sheets=result["detected_sheets"],
            skipped_sheets=result["skipped_sheets"],
        )
    )


if __name__ == "__main__":
    main()
