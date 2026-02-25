"""CLI commands for claude-code-model-gateway."""

import click

from src import __version__


@click.group()
@click.version_option(version=__version__, prog_name="claude-code-model-gateway")
def main():
    """claude-code-model-gateway - A command-line application.

    Use --help on any command for more information.
    """
    pass


@main.command()
def hello():
    """Say hello from claude-code-model-gateway."""
    click.echo("Hello from claude-code-model-gateway!")
    click.echo("Use --help for available commands.")


@main.command()
@click.argument("name", default="World")
def greet(name: str):
    """Greet someone by NAME.

    If no name is provided, greets "World".
    """
    click.echo(f"Hello, {name}!")


@main.command()
def version():
    """Show the application version."""
    click.echo(f"claude-code-model-gateway version {__version__}")


if __name__ == "__main__":
    main()
