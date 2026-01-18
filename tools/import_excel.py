from __future__ import annotations

from pathlib import Path

from config import EXCEL_PATH
from tools.db import get_connection
from tools.importer import debug_workbook_mapping, import_workbook


def _print_mapping_report(sheet_reports: list[dict]) -> None:
    for report in sheet_reports:
        header_row = report.get("header_row")
        header_display = header_row if header_row is not None else "-"
        print(f"Sheet: {report.get('sheet_name')}")
        print(f"  header_row: {header_display}")
        print(
            "  rows: total={total}, inserted={inserted}, skipped={skipped}".format(
                total=report.get("rows_total", 0),
                inserted=report.get("rows_inserted", 0),
                skipped=report.get("rows_skipped", 0),
            )
        )
        print("  mapping:")
        mapping = report.get("mapping") or {}
        for key, value in mapping.items():
            print(f"    - {key}: {value or '-'}")
        unused_headers = report.get("unused_headers") or []
        missing_critical = report.get("missing_critical_fields") or []
        unused_display = ", ".join(unused_headers) if unused_headers else "-"
        missing_display = ", ".join(missing_critical) if missing_critical else "-"
        print(f"  unused_headers: {unused_display}")
        print(f"  missing_critical_fields: {missing_display}")
        print("")


def import_excel(path: str | None = None) -> dict:
    excel_path = Path(path or EXCEL_PATH)
    if not excel_path or not excel_path.exists():
        raise FileNotFoundError("Excel path is not configured or does not exist.")
    conn = get_connection()
    result = import_workbook(excel_path, conn)
    conn.close()
    return result


def reindex_with_report(path: str | None = None) -> dict:
    result = import_excel(path)
    _print_mapping_report(result.get("sheet_reports", []))
    summary = result.get("summary", {})
    if summary:
        print("Reindex summary:")
        print(f"  sheets_detected: {result.get('detected_sheets', 0)}")
        print(f"  rows_scanned: {summary.get('rows_total', 0)}")
        print(f"  rows_inserted: {summary.get('rows_inserted', 0)}")
        print(f"  rows_skipped: {summary.get('rows_skipped', 0)}")
        print(
            "  rows_unit_price_from_unit: "
            f"{summary.get('rows_unit_price_from_unit', 0)}"
        )
        print(
            "  rows_unit_price_from_total_qty: "
            f"{summary.get('rows_unit_price_from_total_qty', 0)}"
        )
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
    result = reindex_with_report(path)
    print(
        "Imported {inserted} rows into DB, detected sheets {detected_sheets}, skipped {skipped_sheets}.".format(
            inserted=result["inserted"],
            detected_sheets=result["detected_sheets"],
            skipped_sheets=result["skipped_sheets"],
        )
    )


def debug_mapping(path: str, *, limit_sheets: int | None = None, max_rows_scan: int = 50) -> None:
    excel_path = Path(path)
    if not excel_path.exists():
        raise FileNotFoundError("Excel path does not exist.")
    result = debug_workbook_mapping(
        excel_path,
        limit_sheets=limit_sheets,
        max_rows_scan=max_rows_scan,
    )
    _print_mapping_report(result.get("sheet_reports", []))


if __name__ == "__main__":
    main()
