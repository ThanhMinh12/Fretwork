import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mcp_server


class MCPAdapterTests(unittest.TestCase):
    def test_chord_retrieval_tool_uses_local_corpus(self):
        result = mcp_server.search_local_chord_knowledge("Dm7", limit=2)
        self.assertTrue(result["local_only"])
        self.assertEqual(result["retrieval"], "local_chord_knowledge")
        self.assertEqual(result["matches"][0]["suffix"], "m7")

    def test_audio_validator_rejects_urls_and_outside_roots(self):
        with self.assertRaises(ValueError):
            mcp_server._resolve_input_file(
                "https://example.com/song.mp3",
                label="Audio file",
                suffixes={".mp3"},
                max_bytes=100,
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "allowed"
            root.mkdir()
            outside = Path(directory) / "outside.mp3"
            outside.write_bytes(b"audio")
            with self.assertRaises(PermissionError):
                mcp_server._resolve_input_file(
                    str(outside),
                    label="Audio file",
                    suffixes={".mp3"},
                    max_bytes=100,
                    roots=(root,),
                )

    def test_report_inspection_is_confined_to_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = output / "take_report.json"
            report.write_text(
                json.dumps({"selected_capo": 2, "workflow_log": ["reviewed"]}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"FRETWORK_MCP_OUTPUT_ROOT": str(output)}):
                result = mcp_server.inspect_transcription_report(
                    str(report), include_workflow_log=False
                )
            self.assertEqual(result["report"]["selected_capo"], 2)
            self.assertNotIn("workflow_log", result["report"])


@unittest.skipIf(mcp_server.mcp is None, "official MCP SDK is optional")
class MCPProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_v2_client_discovers_and_calls_local_retrieval_tool(self):
        from mcp import Client

        async with Client(mcp_server.mcp) as client:
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            result = await client.call_tool(
                "search_local_chord_knowledge", {"query": "Dm7", "limit": 2}
            )
        self.assertTrue(
            {
                "transcribe_local_audio",
                "search_local_chord_knowledge",
                "inspect_transcription_report",
            }.issubset(names)
        )
        self.assertEqual(
            result.structured_content["matches"][0]["id"], "minor-seventh"
        )


if __name__ == "__main__":
    unittest.main()
