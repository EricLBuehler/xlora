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
        "disable_scalings_logging",
        "enable_scalings_logging",
        "flush_log_scalings",
        "from_pretrained",
        "get_nb_trainable_parameters",
        "mole",
        "mole_classifier",
        "mole_config",
        "mole_insertion",
        "mole_state",
        "print_scalings_predictions",
        "print_trainable_parameters",
        "set_scalings_with_lifetime",
    ]
