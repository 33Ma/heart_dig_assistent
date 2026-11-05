"""Local demonstration authentication for the login UI."""

DEMO_ACCOUNT = "admin"
DEMO_PASSWORD = "123456"


def authenticate(account: str, password: str) -> bool:
    """Validate the local demonstration account.

    This function intentionally has no persistence or password storage.
    Replace it with the real authentication service when available.
    """

    return account.strip() == DEMO_ACCOUNT and password == DEMO_PASSWORD
