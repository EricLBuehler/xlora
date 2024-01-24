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
        "disable_scalings_logging",
        "enable_scalings_logging",
        "flush_log_scalings",
        "from_pretrained",
        "get_nb_trainable_parameters",
        "print_scalings_predictions",
        "print_trainable_parameters",
        "set_scalings_with_lifetime",
        "xLoRAConfig",
        "xlora",
        "xlora_classifier",
        "xlora_config",
        "xlora_insertion",
        "xlora_state",
    ]
