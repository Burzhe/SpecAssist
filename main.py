from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="SpecAssist entrypoint")
    parser.add_argument("--profile", action="store_true", help="Profile Excel and exit")
    parser.add_argument("--reindex", action="store_true", help="Recreate DB and import Excel")
    parser.add_argument("--sample", type=int, help="Print first N records from sqlite")
    args = parser.parse_args()

    if args.profile:
        from tools.profile_excel import main as profile_main

        profile_main()
        return

    if args.reindex:
        from tools.import_excel import main as import_main

        import_main(recreate=True)
        return

    if args.sample is not None:
        from tools.import_excel import sample_items

        sample_items(args.sample)
        return

    print("No CLI action specified.")


if __name__ == "__main__":
    main()
