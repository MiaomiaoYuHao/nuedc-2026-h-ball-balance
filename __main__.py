"""允许 `python -m <包名>` 直接启动。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())