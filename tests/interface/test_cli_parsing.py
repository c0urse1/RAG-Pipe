import json
from click.testing import CliRunner

from bu_superagent.interface.cli.main import cli


def test_cli_query_parsing():
    runner = CliRunner()
    result = runner.invoke(cli, ["query", "--text", "hello", "--top-k", "1"])
    assert result.exit_code == 0
    # Should output JSON list
    data = json.loads(result.output)
    assert isinstance(data, list)
