# GitHub Operations Guide

This guide explains how CodeEvolver uses GitHub App authentication to access private repositories.

**Note**: CodeEvolver uses direct GitHub REST API calls (via `httpx` and `pyjwt`) instead of PyGithub to avoid LGPL licensing issues. All dependencies are MIT/Apache-2.0 licensed.

## Service Architecture

CodeEvolver uses two complementary services for GitHub operations:

### `github_app.py` - Authentication Layer
Handles GitHub App authentication via REST API:
- Generates JWT tokens for app authentication
- Retrieves installation access tokens from GitHub's API
- Converts repository URLs to authenticated format
- Verifies installation access to repositories

### `git_service.py` - Git Operations Layer
Performs local git operations using authenticated access:
- Clones repositories (uses `github_app.py` for private repo authentication)
- Creates and manages git worktrees for parallel code mutations
- Commits changes locally
- Manages workspace directories

**How they work together**: When cloning a private repository, `git_service.py` calls `github_app.py` to get an installation token, then uses that token to authenticate the clone operation. All subsequent git operations (worktrees, commits) happen locally using the cloned repository.

## GitHub App Setup

A GitHub App was created in your GitHub organization/account (Settings → Developer settings → GitHub Apps). The app provides an **App ID** and a **Private Key** (PEM format), which are used for authentication. The app must be installed on the repositories you want to access, which provides an **Installation ID** for each installation.

## Configuration

Set the following environment variables:

```bash
export CODEEVOLVER_GITHUB_APP_ID="2671751"
export CODEEVOLVER_GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----"
```


## Testing

### Unit Tests

Run the unit tests (no GitHub credentials needed):

```bash
pytest tests/test_github_app_auth.py::TestGitHubAppService -v
pytest tests/test_github_app_auth.py::TestGitServiceWithAuth -v
```

### Integration Tests

For full integration tests with a real private repository:

1. Set up environment variables:
```bash
export CODEEVOLVER_GITHUB_APP_ID="your_app_id"
export CODEEVOLVER_GITHUB_APP_PRIVATE_KEY="your_private_key"
export GITHUB_TEST_INSTALLATION_ID="your_installation_id"
export GITHUB_TEST_REPO_URL="https://github.com/owner/test-private-repo"
```

2. Run integration tests:
```bash
# Test GitHub App authentication
pytest tests/test_github_app_auth.py::TestGitHubAppIntegration -v -m integration

# Test private repo cloning
pytest tests/test_git_worktree.py::TestGitWorktreeIntegration::test_clone_private_repo_and_create_worktree -v
```

## Additional Resources

- [GitHub Apps Documentation](https://docs.github.com/en/apps)
- [GitHub App Permissions](https://docs.github.com/en/apps/creating-github-apps/setting-up-a-github-app/creating-a-github-app#choosing-permissions)
