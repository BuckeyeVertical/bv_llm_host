# Public access

Serve a vLLM model at a fixed HTTPS URL that anyone with the API key can use, with no OSC
account, VPN or Tailscale on their side. Built on Tailscale Funnel, running in the job with
no root needed. Ollama is never exposed, because it has no authentication.

```bash
EXPOSE=funnel scripts/serve.sh qwen3.6-35b-fp8 --partition=quad
```

The log prints `public: https://<TS_HOSTNAME>.<tailnet>.ts.net/v1` when it is reachable, and
`status.sh` shows the URL too. Clients send the key as an OpenAI bearer token
(`OPENAI_API_KEY`, or `api_key=` in the SDK).

## One-time setup

| Piece | Where |
|---|---|
| Tailscale account | MagicDNS, HTTPS and Funnel turned on in the admin console |
| Static `tailscale` + `tailscaled` binaries | `TAILSCALE_DIR`, from pkgs.tailscale.com |
| `~/.bv_llm_host/secrets.env` on OSC, chmod 600 | `TS_AUTHKEY=tskey-auth-...` and `VLLM_API_KEY=$(openssl rand -hex 24)` |

The first job registers machine `TS_HOSTNAME`. Turn off key expiry for it in the admin
console, or the URL stops working when the key expires. A brand-new name can take a few
minutes before public DNS and its certificate work.

## How it behaves

- **Same URL every job.** The Tailscale identity is kept in `~/.bv_llm_host/tailscale`. The URL
  changes only if that directory is deleted, `TS_HOSTNAME` changes, or the tailnet is renamed.
  If you delete the directory, delete the old machine in the console first, or the new one
  becomes `<name>-1`.
- **One holder at a time.** A second public job serves privately until the first lets go of the
  URL (`~/.bv_llm_host/funnel.owner`), then takes it over.
- **The key is the only protection.** Share it privately. To rotate it, edit `secrets.env`,
  restart the job and tell your users.

## Staying up past walltime

```bash
EXPOSE=funnel AUTO_RESUBMIT=1 scripts/serve.sh qwen3.6-35b-fp8 --partition=debug-quad
```

Ten minutes before walltime the job queues a successor with the same settings. The successor
loads the model while the old job keeps serving, then cancels it and takes over the URL.
`scripts/stop.sh <name>` ends the chain.

- A handoff costs about 1-2 min of downtime when a second GPU is free. If the successor has to
  wait for the same GPU (seen on Ascend `debug-quad`), expect 5-8 min each hour.
- Debug queues allow 2 running jobs per user. Keep the second slot free for the handoff.
- For an event, a regular-queue job long enough to cover it avoids handoffs. Submit it days
  ahead: those queues can take that long to start.
- The chain keeps billing until you stop it.
