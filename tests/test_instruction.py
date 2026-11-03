"""Turning a typed instruction into a checked plan, with a fake vision model."""

from pathlib import Path

import numpy as np
import pytest

from src.reasoning.instruction import (
    InstructionError,
    Order,
    Task,
    choose_glasses,
    parse_plan,
)
from src.reasoning.vlm_client import load_api_key, plan_from_instruction

# Robot frame: x is forward from the robot, y is to its (and the person's) left.
STARTS = [np.array((0.45, 0.10)), np.array((0.30, -0.20)), np.array((0.38, 0.25))]
FLIP_STARTS = [*STARTS, np.array((0.42, -0.05))]


def _reply(task: str = "triangle", order: str = "nearest_first", understood: bool = True) -> str:
    flag = "true" if understood else "false"
    return (
        f'Here is the plan: {{"task": "{task}", "order": "{order}", '
        f'"understood": {flag}, "reason": "the instruction asks for it"}}'
    )


def test_a_reply_with_text_around_the_json_is_read() -> None:
    plan = parse_plan(_reply())

    assert plan.task is Task.TRIANGLE
    assert plan.order is Order.NEAREST_FIRST


def test_a_reply_with_an_unknown_order_is_refused() -> None:
    with pytest.raises(InstructionError):
        parse_plan(_reply(order="tallest_first"))


def test_a_reply_with_no_json_is_refused() -> None:
    with pytest.raises(InstructionError):
        parse_plan("I think the robot should stack them.")


def test_nearest_first_puts_the_farthest_glass_on_top() -> None:
    order, flip = choose_glasses(STARTS, parse_plan(_reply()), video_has_flip=False)

    assert order == (1, 2, 0)
    assert flip is False


def test_farthest_first_reverses_it() -> None:
    order, _ = choose_glasses(
        STARTS, parse_plan(_reply(order="farthest_first")), video_has_flip=False
    )

    assert order == (0, 2, 1)


def test_left_first_uses_the_persons_left() -> None:
    order, _ = choose_glasses(
        STARTS, parse_plan(_reply(order="left_first")), video_has_flip=False
    )

    assert order == (2, 0, 1)


def test_the_video_order_is_kept_when_asked() -> None:
    order, _ = choose_glasses(
        STARTS, parse_plan(_reply(order="as_in_video")), video_has_flip=False
    )

    assert order == (0, 1, 2)


def test_a_flip_needs_a_video_that_shows_a_fourth_glass() -> None:
    with pytest.raises(InstructionError, match="fourth"):
        choose_glasses(STARTS, parse_plan(_reply(task="flipped_tower")), video_has_flip=False)


def test_a_flip_video_can_still_be_asked_for_a_triangle() -> None:
    order, flip = choose_glasses(FLIP_STARTS, parse_plan(_reply()), video_has_flip=True)

    assert order == (1, 2, 0)
    assert flip is False


def test_the_order_also_chooses_which_glass_is_flipped() -> None:
    order, flip = choose_glasses(
        FLIP_STARTS, parse_plan(_reply(task="flipped_tower")), video_has_flip=True
    )

    assert order == (1, 2, 3, 0)
    assert flip is True


def test_keeping_the_video_order_flips_the_glass_the_person_flipped() -> None:
    order, _ = choose_glasses(
        FLIP_STARTS,
        parse_plan(_reply(task="flipped_tower", order="as_in_video")),
        video_has_flip=True,
    )

    assert order == (0, 1, 2, 3)


def test_an_instruction_the_model_did_not_understand_is_refused() -> None:
    with pytest.raises(InstructionError, match="understand"):
        choose_glasses(STARTS, parse_plan(_reply(understood=False)), video_has_flip=False)


class FakeModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.image_counts: list[int] = []

    def ask(self, prompt: str, images: list[bytes]) -> str:
        self.prompts.append(prompt)
        self.image_counts.append(len(images))
        return self.reply


def test_the_instruction_and_pictures_are_sent_and_the_reply_checked() -> None:
    model = FakeModel(_reply(order="farthest_first"))

    plan = plan_from_instruction(model, "farthest at the bottom", [b"a", b"b"], 3, False)

    assert plan.order is Order.FARTHEST_FIRST
    assert "farthest at the bottom" in model.prompts[0]
    assert "3 glasses" in model.prompts[0]
    assert model.image_counts == [2]


def test_a_refused_request_is_a_clear_error_that_hides_the_key(monkeypatch) -> None:
    import io
    import urllib.error
    import urllib.request

    from src.reasoning.vlm_client import AnthropicVision, VlmConfig

    def refuse(*args, **kwargs):
        body = io.BytesIO(b'{"error": "credit balance is too low for key sk-ant-secret"}')
        raise urllib.error.HTTPError("url", 400, "Bad Request", {}, body)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    client = AnthropicVision(VlmConfig("model", 10, 5.0, 1, 100), "sk-ant-secret")

    with pytest.raises(InstructionError) as caught:
        client.ask("hello", [])

    assert "credit balance is too low" in str(caught.value)
    assert "sk-ant-secret" not in str(caught.value)


def test_the_key_is_read_from_an_env_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nANTHROPIC_API_KEY= sk-ant-test \n")

    assert load_api_key(env) == "sk-ant-test"


def test_a_missing_key_is_a_clear_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(InstructionError, match="ANTHROPIC_API_KEY"):
        load_api_key(tmp_path / ".env")
