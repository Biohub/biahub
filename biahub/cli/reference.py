"""The command line as the CLI reference (docs/cli.md) documents it.

mkdocs-typer2 documents every command, hidden ones included. The deprecated registration
aliases and `convert-settings` are hidden from `biahub --help`, and mkdocs-click left
them out of the reference, so this drops them before the reference is built.
"""

from typer.main import get_command

from biahub.cli.main import app

biahub = get_command(app)
for name in [name for name, command in biahub.commands.items() if command.hidden]:
    del biahub.commands[name]
