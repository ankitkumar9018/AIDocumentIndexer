"""
CLI commands for Mandala Sync.
"""

import getpass
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .api import APIClient, APIError
from .config import (
    add_watched_directory,
    clear_token,
    get_server_url,
    get_settings,
    get_token,
    get_watched_directories,
    remove_watched_directory,
    set_server_url,
    set_setting,
    set_token,
)
from .watcher import LocalWatcher

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """Mandala Sync - Sync local folders with Mandala Document Indexer."""
    pass


# =============================================================================
# Authentication Commands
# =============================================================================

@main.command()
@click.option("--server", "-s", help="Server URL (e.g., https://your-server.com)")
@click.option("--email", "-e", help="Email address")
@click.option("--password", "-p", help="Password (will prompt if not provided)")
def login(server: str, email: str, password: str):
    """Login to Mandala server and store credentials."""
    if server:
        set_server_url(server)

    server_url = get_server_url()
    if not server_url:
        server_url = click.prompt("Server URL")
        set_server_url(server_url)

    if not email:
        email = click.prompt("Email")

    if not password:
        password = getpass.getpass("Password: ")

    try:
        with console.status("Logging in..."):
            client = APIClient(server_url=server_url)
            result = client.login(email, password)

        token = result.get("access_token")
        if not token:
            console.print("[red]Login failed: No token received[/red]")
            sys.exit(1)

        set_token(token)
        console.print(f"[green]Successfully logged in as {email}[/green]")

    except APIError as e:
        console.print(f"[red]Login failed: {e.detail or e.message}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
def logout():
    """Clear stored credentials."""
    clear_token()
    console.print("[green]Logged out successfully[/green]")


@main.command()
def whoami():
    """Show current login status."""
    server = get_server_url()
    token = get_token()

    if not server:
        console.print("[yellow]Not configured. Run: mandala-sync login[/yellow]")
        return

    console.print(f"Server: {server}")

    if not token:
        console.print("[yellow]Not logged in[/yellow]")
        return

    try:
        with APIClient() as client:
            user = client.verify_token()
            console.print(f"Logged in as: {user.get('email', 'Unknown')}")
            console.print(f"Name: {user.get('name', 'N/A')}")
    except APIError as e:
        console.print(f"[red]Token invalid: {e.message}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# =============================================================================
# Directory Commands
# =============================================================================

@main.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
@click.option("--recursive/--no-recursive", default=True, help="Watch subdirectories")
@click.option("--collection", "-c", help="Collection name for uploaded files")
@click.option("--tier", "-t", type=int, default=1, help="Access tier (1-5)")
@click.option("--folder", "-f", help="Target folder ID")
def watch(path: str, recursive: bool, collection: str, tier: int, folder: str):
    """Add a directory to watch for new files."""
    try:
        dir_config = add_watched_directory(
            path=path,
            recursive=recursive,
            collection=collection,
            access_tier=tier,
            folder_id=folder,
        )
        console.print(f"[green]Now watching: {dir_config['path']}[/green]")

        if recursive:
            console.print("  Recursive: Yes")
        if collection:
            console.print(f"  Collection: {collection}")
        console.print(f"  Access Tier: {tier}")

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)


@main.command()
@click.argument("path", type=click.Path())
def unwatch(path: str):
    """Remove a directory from watch list."""
    resolved = str(Path(path).resolve())

    if remove_watched_directory(resolved):
        console.print(f"[green]Stopped watching: {resolved}[/green]")
    else:
        console.print(f"[yellow]Directory not in watch list: {resolved}[/yellow]")


@main.command(name="list")
def list_directories():
    """List watched directories."""
    directories = get_watched_directories()

    if not directories:
        console.print("[yellow]No directories being watched[/yellow]")
        console.print("Add one with: mandala-sync watch /path/to/folder")
        return

    table = Table(title="Watched Directories")
    table.add_column("Path", style="cyan")
    table.add_column("Recursive")
    table.add_column("Collection")
    table.add_column("Tier")
    table.add_column("Enabled")

    for d in directories:
        table.add_row(
            d["path"],
            "Yes" if d.get("recursive", True) else "No",
            d.get("collection") or "-",
            str(d.get("access_tier", 1)),
            "[green]Yes[/green]" if d.get("enabled", True) else "[red]No[/red]",
        )

    console.print(table)


# =============================================================================
# Watcher Commands
# =============================================================================

@main.command()
@click.option("--scan/--no-scan", default=False, help="Scan directories for existing files")
def start(scan: bool):
    """Start watching directories and uploading files."""
    token = get_token()
    if not token:
        console.print("[red]Not logged in. Run: mandala-sync login[/red]")
        sys.exit(1)

    directories = get_watched_directories()
    if not directories:
        console.print("[yellow]No directories to watch. Add one with: mandala-sync watch /path[/yellow]")
        sys.exit(1)

    try:
        api_client = APIClient()
        api_client.verify_token()
    except APIError:
        console.print("[red]Invalid token. Run: mandala-sync login[/red]")
        sys.exit(1)

    watcher = LocalWatcher(api_client=api_client)

    # Initial scan if requested
    if scan:
        console.print("Scanning directories for existing files...")
        for d in directories:
            if d.get("enabled", True):
                count = watcher.scan_directory(d["path"], d)
                console.print(f"  Found {count} files in {d['path']}")

    watcher.start()

    console.print("\n[green]Watching for new files. Press Ctrl+C to stop.[/green]\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/yellow]")
        watcher.stop()
        api_client.close()


@main.command()
def status():
    """Show current watcher status."""
    directories = get_watched_directories()
    server = get_server_url()
    token = get_token()

    console.print(f"\n[bold]Mandala Sync Status[/bold]\n")

    console.print(f"Server: {server or '[yellow]Not configured[/yellow]'}")
    console.print(f"Logged in: {'[green]Yes[/green]' if token else '[red]No[/red]'}")
    console.print(f"Directories: {len(directories)}")

    if directories:
        console.print("\nWatched directories:")
        for d in directories:
            status_icon = "[green]●[/green]" if d.get("enabled", True) else "[red]○[/red]"
            console.print(f"  {status_icon} {d['path']}")


@main.command()
@click.argument("path", type=click.Path(exists=True, file_okay=False))
def scan(path: str):
    """Scan a directory and upload existing files."""
    token = get_token()
    if not token:
        console.print("[red]Not logged in. Run: mandala-sync login[/red]")
        sys.exit(1)

    resolved = str(Path(path).resolve())

    # Find matching config
    directories = get_watched_directories()
    dir_config = None
    for d in directories:
        if d["path"] == resolved:
            dir_config = d
            break

    if not dir_config:
        console.print(f"[yellow]Directory not in watch list. Using defaults.[/yellow]")
        dir_config = {
            "path": resolved,
            "recursive": True,
            "collection": None,
            "access_tier": 1,
            "folder_id": None,
        }

    try:
        api_client = APIClient()
        api_client.verify_token()
    except APIError:
        console.print("[red]Invalid token. Run: mandala-sync login[/red]")
        sys.exit(1)

    watcher = LocalWatcher(api_client=api_client)

    console.print(f"Scanning {resolved}...")
    count = watcher.scan_directory(resolved, dir_config)
    console.print(f"Found {count} files")

    if count > 0:
        console.print("Starting upload...")
        watcher.start()

        try:
            while watcher._upload_queue.qsize() > 0:
                time.sleep(1)
            time.sleep(2)  # Wait for last upload to complete
        except KeyboardInterrupt:
            pass

        watcher.stop()

    api_client.close()
    console.print("[green]Done[/green]")


# =============================================================================
# Config Commands
# =============================================================================

@main.group()
def config():
    """Manage configuration settings."""
    pass


@config.command(name="get")
@click.argument("key", required=False)
def config_get(key: str):
    """Get a configuration value."""
    settings = get_settings()

    if key:
        value = settings.get(key)
        if value is not None:
            console.print(f"{key} = {value}")
        else:
            console.print(f"[yellow]Setting not found: {key}[/yellow]")
    else:
        for k, v in settings.items():
            console.print(f"{k} = {v}")


@config.command(name="set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value."""
    # Convert value types
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.isdigit():
        value = int(value)

    set_setting(key, value)
    console.print(f"[green]Set {key} = {value}[/green]")


if __name__ == "__main__":
    main()
