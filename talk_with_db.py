"""Compatibility entry point for the original experimental script.

Importing this module is now side-effect free. Run it directly to start the
shared command-line application.
"""

from cli_interface import main


if __name__ == "__main__":
    raise SystemExit(main())
