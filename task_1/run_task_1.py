"""Point d'entree du pipeline LOSO ou de la generation des figures."""
import sys

if __package__:
    from . import loso_normalisation_sujet, visualiser_resultats
else:
    import loso_normalisation_sujet
    import visualiser_resultats


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "visualiser":
        return visualiser_resultats.main(args[1:])
    if args and args[0] == "experiment":
        return loso_normalisation_sujet.main(args[1:])
    if args and args[0] not in ("-h", "--help"):
        raise SystemExit("Commande attendue : experiment ou visualiser.")
    print("usage: python -m task_1.run_task_1 {experiment,visualiser} [options]\n"
          "Utilisez --help apres la commande pour voir ses options.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
