import mole


def test_items():
    assert dir(mole) == [
        "MoLEConfig",
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__path__",
        "__spec__",
        "add_mole_to_model",
        "from_pretrained",
        "mole",
        "mole_classifier",
        "mole_config",
        "mole_insertion_layers",
        "mole_model",
        "mole_state",
        "set_scalings_with_lifetime",
    ]
