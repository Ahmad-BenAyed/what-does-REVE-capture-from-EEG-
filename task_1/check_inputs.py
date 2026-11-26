"""Verifier les chemins des entrees sans lancer de regression."""
import sys

if __package__:
    from .loso_normalisation_sujet import main
else:
    from loso_normalisation_sujet import main

if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--check"]))
