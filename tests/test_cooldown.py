from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from plugin import TarotsPlugin


class CooldownTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.plugin = object.__new__(TarotsPlugin)
        self.plugin._cooldown_file_path = Path(self.temp_dir.name) / "tarot_cooldown.json"
        self.plugin._cooldown_entries = {}
        self.plugin._cooldown_loaded = False
        self.plugin._cooldown_file_lock = asyncio.Lock()
        self.plugin._cooldown_key_locks = {}
        self.plugin._ctx = SimpleNamespace(
            logger=SimpleNamespace(
                warning=MagicMock(),
                debug=MagicMock(),
                error=MagicMock(),
            )
        )
        self.plugin._plugin_config_instance = SimpleNamespace(
            adjustment=SimpleNamespace(
                cooldown_enabled=True,
                cooldown_seconds=3600,
                cooldown_notice_text="刚刚已经占卜过了，过{minutes}分钟再来吧",
            )
        )
        self.runtime = SimpleNamespace(
            execute=AsyncMock(return_value=(True, "ok")),
            _send_after_delay=AsyncMock(return_value=True),
        )
        self.message = {
            "session_id": "stream-a",
            "platform": "qq",
            "message_info": {
                "user_info": {
                    "user_id": "10001",
                    "user_nickname": "user",
                }
            },
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    async def _execute(self, stream_id: str = "stream-a", message: dict | None = None) -> tuple[bool, str]:
        return await self.plugin._execute_tarot_with_cooldown(
            self.runtime,
            stream_id,
            "自动",
            "单张",
            "user",
            "占卜",
            message=self.message if message is None else message,
        )

    async def test_disabled_cooldown_does_not_limit_repeated_requests(self) -> None:
        self.plugin.config.adjustment.cooldown_enabled = False

        await self._execute()
        await self._execute()

        self.assertEqual(self.runtime.execute.await_count, 2)
        self.runtime._send_after_delay.assert_not_awaited()

    async def test_same_user_and_stream_is_blocked_after_success(self) -> None:
        first = await self._execute()
        second = await self._execute()

        self.assertEqual(first, (True, "ok"))
        self.assertEqual(second, (False, "塔罗占卜冷却中"))
        self.assertEqual(self.runtime.execute.await_count, 1)
        self.runtime._send_after_delay.assert_awaited_once()
        self.assertTrue(self.plugin._cooldown_file_path.exists())

    async def test_different_streams_are_independent(self) -> None:
        await self._execute("stream-a")
        await self._execute("stream-b")

        self.assertEqual(self.runtime.execute.await_count, 2)
        self.runtime._send_after_delay.assert_not_awaited()

    async def test_different_users_are_independent(self) -> None:
        other_message = {
            **self.message,
            "message_info": {"user_info": {"user_id": "10002", "user_nickname": "other"}},
        }

        await self._execute(message=self.message)
        await self._execute(message=other_message)

        self.assertEqual(self.runtime.execute.await_count, 2)
        self.runtime._send_after_delay.assert_not_awaited()

    async def test_failed_execution_does_not_set_cooldown(self) -> None:
        self.runtime.execute.return_value = (False, "failed")

        await self._execute()
        await self._execute()

        self.assertEqual(self.runtime.execute.await_count, 2)
        self.runtime._send_after_delay.assert_not_awaited()

    async def test_expired_cooldown_is_cleaned_and_allows_execution(self) -> None:
        self.plugin._cooldown_loaded = True
        self.plugin._cooldown_entries = {"qq|10001|stream-a": 1.0}

        result = await self._execute()

        self.assertEqual(result, (True, "ok"))
        self.assertEqual(self.runtime.execute.await_count, 1)

    async def test_missing_user_id_skips_cooldown(self) -> None:
        message = {"session_id": "stream-a", "platform": "qq", "message_info": {"user_info": {}}}

        await self._execute(message=message)
        await self._execute(message=message)

        self.assertEqual(self.runtime.execute.await_count, 2)
        self.runtime._send_after_delay.assert_not_awaited()

    async def test_concurrent_same_key_allows_only_one_successful_execution(self) -> None:
        async def slow_success(*_args: object, **_kwargs: object) -> tuple[bool, str]:
            await asyncio.sleep(0.01)
            return True, "ok"

        self.runtime.execute.side_effect = slow_success

        results = await asyncio.gather(self._execute(), self._execute())

        self.assertEqual(results.count((True, "ok")), 1)
        self.assertEqual(results.count((False, "塔罗占卜冷却中")), 1)
        self.assertEqual(self.runtime.execute.await_count, 1)
        self.runtime._send_after_delay.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
