from .context import app


def test_app(capsys, example_fixture):
    app.Blueprint.run()
    captured = capsys.readouterr()

    assert "Hello World..." in captured.out
