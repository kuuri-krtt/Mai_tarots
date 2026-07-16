from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from plugin import (
    MEMORY_SILENT_MAX_ENTRIES,
    MEMORY_SILENT_PLACEHOLDER,
    MEMORY_SILENT_TTL_SECONDS,
    TarotRuntime,
    TarotsPlugin,
)


class SendResultTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.plugin = object.__new__(TarotsPlugin)
        self.plugin._memory_silent_texts = {}
        self.plugin._ctx = SimpleNamespace(
            send=SimpleNamespace(
                text=AsyncMock(),
                image=AsyncMock(),
                forward=AsyncMock(),
            ),
            logger=SimpleNamespace(error=MagicMock(), warning=MagicMock()),
        )
        self.plugin._plugin_config_instance = SimpleNamespace(
            plugin=SimpleNamespace(enabled=True),
            adjustment=SimpleNamespace(
                delay_preface_seconds=0,
                delay_image_seconds=0,
                delay_text_seconds=0,
                delay_extension_seconds=0,
                delay_error_seconds=0,
                send_preface=False,
                send_card_names=True,
                send_interpretation=False,
                send_extension_comment=False,
                output_mode="逐条发送",
            )
        )
        self.plugin._bot_display_name = "麦麦"
        self.runtime = TarotRuntime(self.plugin)

    async def test_text_send_false_is_reported_and_silent_marker_is_rolled_back(self) -> None:
        self.plugin.ctx.send.text.return_value = False

        sent = await self.runtime._send_after_delay("text", "测试文本", "stream")

        self.assertFalse(sent)
        self.assertEqual(self.plugin._memory_silent_texts, {})

    async def test_text_send_exception_is_reported_and_silent_marker_is_rolled_back(self) -> None:
        self.plugin.ctx.send.text.side_effect = RuntimeError("send failed")

        sent = await self.runtime._send_after_delay("text", "测试文本", "stream")

        self.assertFalse(sent)
        self.assertEqual(self.plugin._memory_silent_texts, {})

    async def test_text_send_true_keeps_marker_for_before_send_hook(self) -> None:
        self.plugin.ctx.send.text.return_value = True

        sent = await self.runtime._send_after_delay("text", "测试文本", "stream")

        self.assertTrue(sent)
        count, expires_at = self.plugin._memory_silent_texts[("stream", "测试文本")]
        self.assertEqual(count, 1)
        self.assertGreater(expires_at, 0)

    async def test_silent_marker_expires_when_hook_does_not_consume_it(self) -> None:
        with patch("plugin.time.monotonic", return_value=100.0):
            self.plugin._mark_memory_silent_text("stream", "测试文本")

        with patch("plugin.time.monotonic", return_value=100.0 + MEMORY_SILENT_TTL_SECONDS):
            consumed = self.plugin._consume_memory_silent_text("stream", "测试文本")

        self.assertFalse(consumed)
        self.assertEqual(self.plugin._memory_silent_texts, {})

    async def test_silent_marker_table_is_bounded(self) -> None:
        with patch("plugin.time.monotonic", return_value=100.0):
            for index in range(MEMORY_SILENT_MAX_ENTRIES + 500):
                self.plugin._mark_memory_silent_text("stream", f"文本-{index}")

        self.assertEqual(len(self.plugin._memory_silent_texts), MEMORY_SILENT_MAX_ENTRIES)
        self.assertNotIn(("stream", "文本-0"), self.plugin._memory_silent_texts)
        self.assertIn(
            ("stream", f"文本-{MEMORY_SILENT_MAX_ENTRIES + 499}"),
            self.plugin._memory_silent_texts,
        )

    async def test_duplicate_silent_markers_keep_count_until_consumed(self) -> None:
        self.plugin._mark_memory_silent_text("stream", "相同文本")
        self.plugin._mark_memory_silent_text("stream", "相同文本")

        self.assertTrue(self.plugin._consume_memory_silent_text("stream", "相同文本"))
        self.assertTrue(self.plugin._consume_memory_silent_text("stream", "相同文本"))
        self.assertFalse(self.plugin._consume_memory_silent_text("stream", "相同文本"))

    async def test_before_send_uses_nonempty_ephemeral_marker_and_disables_storage(self) -> None:
        self.plugin._mark_memory_silent_text("stream", "塔罗结果")
        message = {
            "session_id": "stream",
            "processed_plain_text": "塔罗结果",
            "raw_message": [{"type": "text", "data": "塔罗结果"}],
        }

        result = await self.plugin.handle_tarots_before_send(
            message=message,
            storage_message=True,
        )

        self.assertIsNotNone(result)
        modified_kwargs = result["modified_kwargs"]
        self.assertEqual(
            modified_kwargs["message"]["processed_plain_text"],
            MEMORY_SILENT_PLACEHOLDER,
        )
        self.assertFalse(modified_kwargs["storage_message"])
        self.assertEqual(
            modified_kwargs["message"]["raw_message"],
            message["raw_message"],
        )

    async def test_before_send_ignores_unregistered_text(self) -> None:
        result = await self.plugin.handle_tarots_before_send(
            message={
                "session_id": "stream",
                "processed_plain_text": "普通回复",
                "raw_message": [{"type": "text", "data": "普通回复"}],
            },
            storage_message=True,
        )

        self.assertIsNone(result)

    async def test_image_send_false_is_not_counted_as_success(self) -> None:
        self.plugin.ctx.send.image.return_value = False
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "card.jpg"
            image_path.write_bytes(b"fake-image")
            self.runtime._find_card_image_path = lambda *_args: image_path

            sent = await self.runtime._send_card_image({"name": "愚者"}, False, "stream")

        self.assertFalse(sent)

    async def test_image_send_true_is_success(self) -> None:
        self.plugin.ctx.send.image.return_value = True
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "card.jpg"
            image_path.write_bytes(b"fake-image")
            self.runtime._find_card_image_path = lambda *_args: image_path

            sent = await self.runtime._send_card_image({"name": "愚者"}, False, "stream")

        self.assertTrue(sent)

    async def test_execute_fails_when_card_image_send_returns_false(self) -> None:
        self.runtime.card_map = {
            "0": {
                "name": "愚者",
                "info": {"description": "开始", "reverseDescription": "鲁莽"},
            }
        }
        self.runtime.formation_map = {"单张": {"cards_num": 1, "is_cut": True, "represent": [["现状"]]}}
        self.runtime._draw_cards = MagicMock(return_value=[("0", False)])
        self.runtime._find_card_image_path = MagicMock(return_value=Path("card.jpg"))
        self.runtime._send_card_image = AsyncMock(return_value=False)
        self.runtime._send_after_delay = AsyncMock(return_value=True)

        with patch("plugin.asyncio.sleep", new=AsyncMock()):
            success, message = await self.runtime.execute("stream")

        self.assertFalse(success)
        self.assertEqual(message, "图片发送失败")

    async def test_execute_fails_when_combined_result_send_returns_false(self) -> None:
        self.runtime.card_map = {
            "0": {
                "name": "愚者",
                "info": {"description": "开始", "reverseDescription": "鲁莽"},
            }
        }
        self.runtime.formation_map = {"单张": {"cards_num": 1, "is_cut": True, "represent": [["现状"]]}}
        self.runtime._draw_cards = MagicMock(return_value=[("0", False)])
        self.runtime._find_card_image_path = MagicMock(return_value=None)
        self.runtime._send_card_image = AsyncMock(return_value=True)
        self.runtime._send_after_delay = AsyncMock(return_value=False)

        with patch("plugin.asyncio.sleep", new=AsyncMock()):
            success, message = await self.runtime.execute("stream")

        self.assertFalse(success)
        self.assertEqual(message, "占卜结果发送失败")

    async def test_card_names_and_interpretation_are_sent_as_one_message(self) -> None:
        self.plugin.config.adjustment.send_interpretation = True
        self.runtime.card_map = {
            "0": {
                "name": "愚者",
                "info": {"description": "新的开始", "reverseDescription": "鲁莽"},
            }
        }
        self.runtime.formation_map = {"单张": {"cards_num": 1, "is_cut": True, "represent": [["现状"]]}}
        self.runtime._draw_cards = MagicMock(return_value=[("0", False)])
        self.runtime._find_card_image_path = MagicMock(return_value=None)
        self.runtime._generate_interpretation = AsyncMock(return_value="保持开放，稳步向前。")
        self.runtime._send_after_delay = AsyncMock(return_value=True)

        with patch("plugin.asyncio.sleep", new=AsyncMock()):
            success, message = await self.runtime.execute("stream")

        self.assertTrue(success)
        self.assertEqual(message, "已抽取塔罗牌")
        self.runtime._send_after_delay.assert_awaited_once_with(
            "text",
            "抽到的牌：\n现状：愚者（正位）\n\n保持开放，稳步向前。",
            "stream",
        )

    async def test_preface_is_built_after_draw_facts_are_prepared(self) -> None:
        self.plugin.config.adjustment.send_preface = True
        self.runtime.card_map = {
            "0": {
                "name": "愚者",
                "info": {"description": "新的开始", "reverseDescription": "鲁莽"},
            }
        }
        self.runtime.formation_map = {"单张": {"cards_num": 1, "is_cut": True, "represent": [["现状"]]}}
        events: list[str] = []

        def draw_cards(card_type: str, formation: str):
            del card_type, formation
            events.append("draw")
            return [("0", False)]

        def find_card_image(card_data, is_reverse):
            del card_data, is_reverse
            events.append("prepare_card")
            return None

        async def build_preface(user, card_type, formation, user_request, card_details):
            self.assertEqual(user, "")
            self.assertEqual(card_type, "当前牌组原生类别（自动）")
            self.assertEqual(formation, "单张")
            self.assertEqual(user_request, "")
            self.assertEqual(len(card_details), 1)
            self.assertEqual(card_details[0]["position"], "现状")
            self.assertEqual(card_details[0]["name"], "愚者")
            self.assertFalse(card_details[0]["is_reverse"])
            self.assertEqual(card_details[0]["description"], "新的开始")
            self.assertTrue(card_details[0]["position_meaning"])
            self.assertEqual(events, ["draw", "prepare_card"])
            events.append("preface")
            return "我先洗牌。"

        async def send_preface(*_args, **_kwargs):
            events.append("send_preface")
            return True

        async def send_text(stage: str, text: str, stream_id: str):
            events.append(f"send_{stage}")
            self.assertEqual(stream_id, "stream")
            self.assertIn("愚者（正位）", text)
            return True

        self.runtime._draw_cards = MagicMock(side_effect=draw_cards)
        self.runtime._find_card_image_path = MagicMock(side_effect=find_card_image)
        self.runtime._build_preface = AsyncMock(side_effect=build_preface)
        self.runtime._send_preface_after_delay = AsyncMock(side_effect=send_preface)
        self.runtime._send_after_delay = AsyncMock(side_effect=send_text)

        with patch("plugin.asyncio.sleep", new=AsyncMock()):
            success, message = await self.runtime.execute("stream")

        self.assertTrue(success)
        self.assertEqual(message, "已抽取塔罗牌")
        self.assertEqual(events, ["draw", "prepare_card", "preface", "send_preface", "send_text"])

    async def test_missing_card_image_continues_with_text_result(self) -> None:
        self.runtime.card_map = {
            "0": {
                "name": "愚者",
                "info": {"description": "开始", "reverseDescription": "鲁莽"},
            }
        }
        self.runtime.formation_map = {"单张": {"cards_num": 1, "is_cut": True, "represent": [["现状"]]}}
        self.runtime._draw_cards = MagicMock(return_value=[("0", False)])
        self.runtime._find_card_image_path = MagicMock(return_value=None)
        self.runtime._send_card_image = AsyncMock(return_value=False)
        self.runtime._send_after_delay = AsyncMock(return_value=True)

        with patch("plugin.asyncio.sleep", new=AsyncMock()):
            success, message = await self.runtime.execute("stream")

        self.assertTrue(success)
        self.assertEqual(message, "已抽取塔罗牌")
        self.runtime._send_card_image.assert_not_awaited()

    async def test_forward_output_sends_one_forward_and_skips_sequential_sends(self) -> None:
        self.plugin.config.adjustment.output_mode = "合并转发"
        self.plugin.config.adjustment.send_preface = True
        self.plugin.ctx.send.forward.return_value = True
        self.runtime.card_map = {
            "0": {
                "name": "愚者",
                "info": {"description": "新的开始", "reverseDescription": "鲁莽"},
            }
        }
        self.runtime.formation_map = {"单张": {"cards_num": 1, "is_cut": False, "represent": [["现状"]]}}
        self.runtime._draw_cards = MagicMock(return_value=[("0", False)])
        self.runtime._build_preface = AsyncMock(return_value="先静一静。")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "card.jpg"
            image_path.write_bytes(b"fake-image")
            self.runtime._find_card_image_path = MagicMock(return_value=image_path)

            success, message = await self.runtime.execute("stream")

        self.assertTrue(success)
        self.assertEqual(message, "已抽取塔罗牌")
        self.plugin.ctx.send.forward.assert_awaited_once()
        self.plugin.ctx.send.text.assert_not_awaited()
        self.plugin.ctx.send.image.assert_not_awaited()
        args, kwargs = self.plugin.ctx.send.forward.await_args
        self.assertEqual(args[1], "stream")
        self.assertFalse(kwargs["storage_message"])
        messages = args[0]
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["segments"][0]["type"], "text")
        self.assertEqual(messages[1]["segments"][0]["type"], "image")
        self.assertEqual(messages[2]["segments"][0]["type"], "text")
        self.assertEqual(messages[2]["segments"][0]["content"], "抽到的牌：\n现状：愚者（正位）")
        self.runtime._build_preface.assert_awaited_once()
        preface_args = self.runtime._build_preface.await_args.args
        self.assertEqual(preface_args[4][0]["name"], "愚者")
        self.assertFalse(preface_args[4][0]["is_reverse"])

    async def test_forward_output_failure_is_reported(self) -> None:
        self.plugin.config.adjustment.output_mode = "合并转发"
        self.plugin.ctx.send.forward.return_value = False
        self.runtime.card_map = {
            "0": {
                "name": "愚者",
                "info": {"description": "新的开始", "reverseDescription": "鲁莽"},
            }
        }
        self.runtime.formation_map = {"单张": {"cards_num": 1, "is_cut": False, "represent": [["现状"]]}}
        self.runtime._draw_cards = MagicMock(return_value=[("0", False)])
        self.runtime._find_card_image_path = MagicMock(return_value=None)

        success, message = await self.runtime.execute("stream")

        self.assertFalse(success)
        self.assertEqual(message, "合并转发发送失败")


if __name__ == "__main__":
    unittest.main()
