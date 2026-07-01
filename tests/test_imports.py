def test_package_modules_can_be_imported():
    """The installed package and its two modules should be importable."""
    import propagation
    import propagation.kemeny

    assert propagation is not None
    assert propagation.kemeny is not None