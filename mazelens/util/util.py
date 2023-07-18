from rich.console import Console
from rich.syntax import Syntax


def print_yaml(yaml):
    console = Console()
    console.print(Syntax(yaml, "yaml"))
