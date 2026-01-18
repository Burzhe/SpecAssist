from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="SpecAssist entrypoint")
    parser.add_argument("--profile", action="store_true", help="Profile Excel and exit")
    parser.add_argument("--reindex", type=str, help="Recreate DB and import Excel from path")
    parser.add_argument("--debug-mapping", type=str, help="Debug Excel sheet mappings")
    parser.add_argument("--limit-sheets", type=int, help="Limit sheets in debug mapping mode")
    parser.add_argument(
        "--max-rows-scan",
        type=int,
        default=50,
        help="Max rows to scan in debug mapping mode (default 50)",
    )
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--sample", type=int, help="Print first N records from sqlite")
    parser.add_argument("--facets", action="store_true", help="Print flag/material counts")
    args = parser.parse_args()

    if args.profile:
        from tools.profile_excel import main as profile_main

        profile_main()
        return

    if args.reindex:
        from tools.import_excel import main as import_main

        import_main(args.reindex)
        return

    if args.debug_mapping:
        from tools.import_excel import debug_mapping

        debug_mapping(
            args.debug_mapping,
            limit_sheets=args.limit_sheets,
            max_rows_scan=args.max_rows_scan,
        )
        return

    if args.search:
        from tools.db import get_connection
        from tools.search import search_items

        conn = get_connection()
        result = search_items(conn, args.search)
        conn.close()
        print(f"Found {len(result['results'])} (tol={result['tol']}, relaxed={result['relaxed']})")
        for item in result["results"]:
            dims = "x".join(
                str(val) for val in (item.get("w_mm"), item.get("d_mm"), item.get("h_mm")) if val
            )
            print(
                f"- {item.get('name') or '-'} | dims={dims or '-'} | price={item.get('price_unit_ex_vat') or '-'} | "
                f"sheet={item.get('source_sheet')} row={item.get('source_row')}"
            )
        return

    if args.sample is not None:
        from tools.import_excel import sample_items

        sample_items(args.sample)
        return

    if args.facets:
        from tools.db import get_connection

        conn = get_connection()
        cursor = conn.execute(
            """
            SELECT
                SUM(has_led) AS has_led,
                SUM(mat_ldsp) AS mat_ldsp,
                SUM(mat_mdf) AS mat_mdf,
                SUM(mat_veneer) AS mat_veneer,
                SUM(has_glass) AS has_glass,
                SUM(has_metal) AS has_metal
            FROM items
            """
        )
        row = cursor.fetchone()
        conn.close()
        print("Facets:", dict(row) if row else {})
        return

    from tools.telegram_bot import run_bot

    run_bot()


if __name__ == "__main__":
    main()
