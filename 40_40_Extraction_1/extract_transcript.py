import json
import os

log_file = r"C:\Users\WangYD\.gemini\antigravity-ide\brain\68eff419-53da-4d04-9ede-0166d325408a\.system_generated\logs\transcript.jsonl"

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            # Look for write_to_file tool calls targeting extraction.py
            if "tool_calls" in data:
                for tc in data["tool_calls"]:
                    if tc.get("name") == "write_to_file":
                        args = tc.get("args", {})
                        target = args.get("TargetFile", "")
                        if "extraction.py" in target:
                            print(f"Found write to {target}:")
                            # Write it to a temporary file so we can view it
                            content = args.get("CodeContent", "")
                            with open("extraction_restored.py", "w", encoding="utf-8") as out:
                                out.write(content)
                            print("Restored file written to extraction_restored.py")
        except Exception as e:
            pass
