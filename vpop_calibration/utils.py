import os

if "IS_PYTEST_RUNNING" in os.environ:
    smoke_test = True
else:
    smoke_test = False
