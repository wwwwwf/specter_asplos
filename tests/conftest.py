"""Apply explicitly configured cache paths before importing test modules."""
import os

if os.environ.get('SPECTER_CONFIG'):
    from configuration import configure
    configure()
