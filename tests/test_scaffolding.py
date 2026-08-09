"""Verifica que el paquete es importable y que la fixture arma el dataset."""

from pathlib import Path


def test_paquetes_importables():
    import src.data
    import src.engine
    import src.modeling
    import src.reporting


def test_dataset_sintetico_tiene_la_estructura_esperada(synthetic_dataset: Path):
    for split, cantidad in [("train", 5), ("val", 2)]:
        images = sorted((synthetic_dataset / split / "images").glob("*.jpg"))
        labels = sorted((synthetic_dataset / split / "labels").glob("*.txt"))
        assert len(images) == cantidad
        assert len(labels) == cantidad

    vacio = synthetic_dataset / "train" / "labels" / "img_2.txt"
    assert vacio.read_text() == ""
