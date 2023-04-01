import pytest
from chatgpt_cli.cli import ChatContext, Message


@pytest.fixture
def config(tmp_path_factory):
    chatlogdir = tmp_path_factory.mktemp("hoge")
    yield {
        "chatlog": {
            "dir": chatlogdir.absolute()
        }
    }


@pytest.fixture
def test_jsons():
    return [
        ("this is a ```[{\"valid\": \"JSON\"}]``` string with additional text.", [{'valid': 'JSON'}]),
        ("this is a ```[{\"valid\": [{\"nested\":\"JSON\"}]}]``` string with additional text.", [{'valid': [{'nested': 'JSON'}]}]),
        ("this is a ```[{\"valid\": [{\"nested\":\"JSON\"},{\"nested\":\"JSON2\"}]}]``` string with additional text.", [{'valid': [{'nested': 'JSON'}, {'nested': 'JSON2'}]}]),
        ("this is a ```[{valid\": [{\"nested\":\"JSON\"},{\"nested\":\"JSON2\"}]}]``` string with additional text.", None),
        ("this i ``[{vSON2\"}]}]``` string with additional text.", None),
    ]


class TestChatContext:
    def test_save_and_resolve_returns_same_messages(self, config):
        title = "some-title"
        chat_context = ChatContext.new(config, title)

        expected: list[Message] = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="World"),
        ]
        chat_context.flush(expected)

        new_chat_context = ChatContext(config=config, logfilepath=chat_context.logfilepath)
        actual = new_chat_context.resolve()

        assert actual == expected

    def test_extract_json(self, config, test_jsons):
        title = "some-title"
        chat_context = ChatContext.new(config, title)

        for string, expected in test_jsons:
            actual = chat_context._extract_json(string)
            assert actual == expected

    def test_new_chatcontext_returns_empty(self, config):
        title = "some-empty-file"
        chat_context = ChatContext.new(config, title)

        assert chat_context.resolve() == []
