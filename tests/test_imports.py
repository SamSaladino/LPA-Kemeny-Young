def test_package_modules_can_be_imported():
    """The installed package and its two modules should be importable."""
    import propagation
    import propagation.lpa
    import propagation.kemeny_young

    assert propagation is not None
    assert propagation.lpa is not None
    assert propagation.kemeny_young is not None