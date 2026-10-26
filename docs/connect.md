# Connecting to the Buckeye Vertical LLM server

The server speaks the OpenAI API, so any OpenAI client works. You need no OSC account,
Tailscale, or VPN: just the URL and the API key.

| | |
|---|---|
| URL | `https://bv-llm.tail43a32e.ts.net/v1` |
| API key | Ask the organizer. Don't share it or commit it to a repo. |
| Model | `Qwen/Qwen3.6-35B-A3B-FP8` |

## 1. Check it works

macOS / Linux:

```bash
export KEY=<the API key>
curl https://bv-llm.tail43a32e.ts.net/v1/models -H "Authorization: Bearer $KEY"
```

Windows (PowerShell):

```powershell
$env:KEY = "<the API key>"
curl.exe https://bv-llm.tail43a32e.ts.net/v1/models -H "Authorization: Bearer $env:KEY"
```

You should get JSON listing the model. Opening the URL in a browser shows `401 Unauthorized`;
that is normal, since browsers don't send the key.

## 2. Use it from Python

```bash
pip install openai
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://bv-llm.tail43a32e.ts.net/v1",
    api_key=os.environ["KEY"],
    max_retries=8,   # rides out the server's brief restarts
    timeout=300,
)
reply = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B-FP8",
    messages=[{"role": "user", "content": "What is a drone?"}],
    max_tokens=2000,
)
print(reply.choices[0].message.content)
```

Tool calling (`tools=[...]`) and images work as in the OpenAI API.

## 3. Use it with bv_llm_sdk

```bash
export BV_AGENT_BACKEND=openai
export BV_AGENT_BASE_URL=https://bv-llm.tail43a32e.ts.net/v1
export BV_AGENT_MODEL=Qwen/Qwen3.6-35B-A3B-FP8
export BV_AGENT_VISION=true
export OPENAI_API_KEY=<the API key>
bv-agent
```

On Windows PowerShell, set each one as `$env:BV_AGENT_BACKEND = "openai"` and so on.

## Tips

- **Thinking.** The model reasons before answering, which makes it more accurate but slower.
  For quick answers, pass `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`
  in Python. Keep it on for flying the drone.
- **Give it room.** If a reply comes back empty, it used up `max_tokens` while thinking.
  Raise `max_tokens`.
- **It's shared.** Replies take a few seconds, longer when many people are using it.

## Troubleshooting

| What you see | What it means |
|---|---|
| `401 Unauthorized` | Key missing or mistyped |
| Connection error or `502` for a minute or two | The server is restarting; retry (`max_retries` does this) |
| Can't resolve or connect for more than ~10 min | The server is down; tell the organizer |
| `404` mentioning the model | Model name typo, or the backup model is running: check `/v1/models` |
| Empty reply | Raise `max_tokens` (see Tips) |
