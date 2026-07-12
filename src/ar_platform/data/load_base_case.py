"""Load the committed base-case CSVs into the MySQL database.

    PYTHONPATH=src python -m ar_platform.data.load_base_case [--force]

Idempotent: does nothing if the database already has data, unless --force is
passed (which wipes working state and reloads the pristine base case).
"""

from __future__ import annotations

import argparse

from ar_platform.data.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the base case into MySQL.")
    parser.add_argument(
        "--force", action="store_true", help="wipe working state and reload"
    )
    args = parser.parse_args()

    with Store() as store:
        store.load_base_case(force=args.force)
        n_cust = len(store.get_customers())
        n_inv = len(store.get_invoices())
        print(f"Base case loaded into {store.db_url}")
        print(f"  customers: {n_cust}\n  invoices:  {n_inv}")


if __name__ == "__main__":
    main()
