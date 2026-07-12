-- Runs once on first MySQL container boot (docker-entrypoint-initdb.d).
-- The main `ar_platform` DB + `ar_user` are created via compose env vars;
-- here we add the separate test database and grant the app user access.
CREATE DATABASE IF NOT EXISTS ar_platform_test
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
GRANT ALL PRIVILEGES ON ar_platform_test.* TO 'ar_user'@'%';
FLUSH PRIVILEGES;
