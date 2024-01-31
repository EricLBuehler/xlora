import xlora


def test_items():
    assert dir(xlora) == [
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "add_xlora_to_model",
        "from_pretrained",
        "load_scalings_log",
        "xLoRAConfig",
        "xlora",
        "xlora_classifier",
        "xlora_config",
        "xlora_insertion",
    ]
