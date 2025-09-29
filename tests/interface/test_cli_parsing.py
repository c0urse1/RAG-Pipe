from bu_superagent.interface.cli.main import main


def test_cli_main_placeholder():
    try:
        main()
    except NotImplementedError:
        assert True
