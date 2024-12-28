import os

import pytest

mark_slow = pytest.mark.skipif(not os.getenv("RUNSLOW"), reason="slow")
