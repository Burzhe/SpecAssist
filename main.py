from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="SpecAssist entrypoint")
    parser.add_argument("--profile", action="store_true", help="Profile Excel and exit")
    args = parser.parse_args()

    if args.profile:
        from tools.profile_excel import main as profile_main

        profile_main()
        return

    print("No CLI action specified.")


if __name__ == "__main__":
    main()
