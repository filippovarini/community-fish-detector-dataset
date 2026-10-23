"""
Dryad (datadryad.org) downloads. The web download links sit behind a bot check, so files are
fetched through the Dryad API with the user's credentials: DRYAD_CLIENT_ID + DRYAD_CLIENT_SECRET
(an API account, from which a fresh token is requested on each run) or DRYAD_TOKEN (a token
copied from the account page, valid for 10 hours), set in the environment or in the repo's .env.
"""

import os
from pathlib import Path

import requests

DRYAD_TOKEN_URL = "https://datadryad.org/oauth/token"


def get_dryad_file_download_url(file_id: int) -> str:
    return f"https://datadryad.org/api/v2/files/{file_id}/download"


def read_dryad_credentials() -> dict:
    """DRYAD_* variables from the environment, falling back to the repo's .env file"""
    credentials = {}
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().startswith("DRYAD_"):
                credentials[key.strip()] = value.strip().strip("\"'")
    credentials.update({k: v for k, v in os.environ.items() if k.startswith("DRYAD_")})
    return credentials


def has_dryad_credentials(credentials: dict) -> bool:
    return "DRYAD_TOKEN" in credentials or (
        "DRYAD_CLIENT_ID" in credentials and "DRYAD_CLIENT_SECRET" in credentials
    )


def get_dryad_download_request(file_id: int, credentials: dict):
    """
    Returns a function that gives (url, headers) for a Dryad file, for HttpRangeFile: a fresh
    bearer token is used to resolve the API download link to its signed S3 URL, and the function
    is called again when that URL expires.
    """
    download_url = get_dryad_file_download_url(file_id)

    def get_token() -> str:
        if "DRYAD_CLIENT_ID" in credentials and "DRYAD_CLIENT_SECRET" in credentials:
            response = requests.post(DRYAD_TOKEN_URL, data={
                "client_id": credentials["DRYAD_CLIENT_ID"],
                "client_secret": credentials["DRYAD_CLIENT_SECRET"],
                "grant_type": "client_credentials",
            }, timeout=60)
            response.raise_for_status()
            return response.json()["access_token"]
        return credentials["DRYAD_TOKEN"]

    def get_request():
        headers = {"Authorization": f"Bearer {get_token()}"}
        response = requests.get(download_url, headers=headers, allow_redirects=False, timeout=60)
        if response.is_redirect:
            return response.headers["Location"], {}
        response.raise_for_status()
        return download_url, headers

    return get_request


def get_dryad_credentials_help(data_page_url: str, zip_path: Path, used_part: str) -> str:
    """Instructions printed when there are neither Dryad credentials nor a local zip"""
    return f"""
Dryad only serves files to signed-in browsers or to API clients with a token. Either:

  1. Use the Dryad API (only {used_part} is downloaded, not the whole zip):
     - Sign in to https://datadryad.org with ORCID, go to My account > API and create an
       API account
     - Put its credentials in the repo's .env file (git-ignored) or in the environment:
         DRYAD_CLIENT_ID=...
         DRYAD_CLIENT_SECRET=...
       (or DRYAD_TOKEN=<a token copied from the account page>, valid for 10 hours)

  2. Or download the whole {zip_path.name} manually from
       {data_page_url}
     and move it to {zip_path}
     The script picks it up automatically and extracts what it needs from it.

Then rerun this script."""
