"""Delete codeevolver branches for a given run timestamp via the /cleanup-branches endpoint."""

import argparse
import sys
from pathlib import Path

import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import determine_api_url


def main():
    parser = argparse.ArgumentParser(
        description="Delete remote codeevolver-* branches for a specific run timestamp.",
    )
    parser.add_argument(
        "repo_url",
        help="GitHub repository URL (e.g., https://github.com/user/repo)",
    )
    parser.add_argument(
        "date",
        help="Run timestamp in YYYYMMDDHHmmss format (e.g., 20260221191443), or 'none' to match all codeevolver branches",
    )
    parser.add_argument(
        "--except",
        dest="except_branches",
        nargs="*",
        default=[],
        help="Branch names to keep (space-separated). Pass at least one or use --except with no value for an empty list.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which branches would be deleted without actually deleting them.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override API base URL (defaults to environment-based URL)",
    )
    args = parser.parse_args()

    # Validate date format locally before making the request
    is_none = args.date.lower() == "none"
    if not is_none and (len(args.date) != 14 or not args.date.isdigit()):
        print(f"Error: date must be in YYYYMMDDHHmmss format (14 digits) or 'none' for all dates. Got: {args.date!r}")
        sys.exit(1)

    # Normalize for the API
    date_value = "none" if is_none else args.date
    prefix = "codeevolver-" if is_none else f"codeevolver-{args.date}"

    base_url = args.url or determine_api_url()
    endpoint = f"{base_url}/cleanup-branches"

    payload = {
        "repo_url": args.repo_url,
        "date": date_value,
        "except_branches": args.except_branches,
    }

    if args.dry_run:
        print(f"[DRY RUN] Would call POST {endpoint}")
        print(f"  repo_url: {args.repo_url}")
        print(f"  date:     {date_value}")
        print(f"  prefix:   {prefix}")
        print(f"  except:   {args.except_branches}")
        print("\nTo execute, run again without --dry-run.")
        return

    print(f"Calling POST {endpoint}")
    print(f"  Deleting branches matching: {prefix}*")
    if args.except_branches:
        print(f"  Keeping: {args.except_branches}")

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(endpoint, json=payload)

        if resp.status_code != 200:
            print(f"\nError: HTTP {resp.status_code}")
            print(resp.text)
            sys.exit(1)

        data = resp.json()
        deleted = data.get("deleted", [])
        skipped = data.get("skipped", [])
        errors = data.get("errors", [])

        print(f"\nDeleted ({len(deleted)}):")
        for b in deleted:
            print(f"  - {b}")

        if skipped:
            print(f"\nSkipped / kept ({len(skipped)}):")
            for b in skipped:
                print(f"  - {b}")

        if errors:
            print(f"\nErrors ({len(errors)}):")
            for e in errors:
                print(f"  - {e}")

        if not deleted and not skipped and not errors:
            print("\nNo branches found matching the pattern.")

    except httpx.RequestError as e:
        print(f"\nConnection error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
