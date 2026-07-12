# Database setup (MySQL 8)

The platform stores its ledger in MySQL. You need a running MySQL 8 server and a
dedicated database + user. Two ways to get there:

## Option A — Docker (recommended, self-contained)

```bash
docker compose up
```

This starts a `mysql:8.0` service with the `ar_platform` database, the
`ar_user` account, and (via `docs/mysql-init/`) the `ar_platform_test` database
for the test suite — no manual steps.

## Option B — Existing local MySQL server

Create the databases and app user once (run in MySQL Workbench or the `mysql`
CLI as an admin):

```sql
CREATE DATABASE IF NOT EXISTS ar_platform      CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE IF NOT EXISTS ar_platform_test CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'ar_user'@'localhost' IDENTIFIED BY 'ar_password';
GRANT ALL PRIVILEGES ON ar_platform.*      TO 'ar_user'@'localhost';
GRANT ALL PRIVILEGES ON ar_platform_test.* TO 'ar_user'@'localhost';
FLUSH PRIVILEGES;
```

Then point the app at it (defaults already match the above):

```bash
export AR_DB_URL=mysql+pymysql://ar_user:ar_password@localhost:3306/ar_platform
export AR_TEST_DB_URL=mysql+pymysql://ar_user:ar_password@localhost:3306/ar_platform_test
```

## Loading the base case

The schema is created automatically on first connection. Load the committed
base-case ledger into MySQL with:

```bash
PYTHONPATH=src python -m ar_platform.data.load_base_case
```

Or just run the dashboard / simulation — both load the base case on startup.

## Notes

- The `ar_user` password (`ar_password`) is a local-dev default. Change it and
  set `AR_DB_URL` accordingly; real credentials belong in `.env` (gitignored).
- PyMySQL needs the `cryptography` package for MySQL 8's default
  `caching_sha2_password` auth — it's in `requirements.txt`.
