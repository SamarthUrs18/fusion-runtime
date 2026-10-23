# Phone simulator

Call a fusion agent the way a phone call through Twilio does, from your own machine: no Twilio
account, no phone number, no phone.

It does what Twilio does when someone calls a number pointed at fusion:

1. posts the incoming-call webhook, signed with the auth token (`X-Twilio-Signature`)
2. reads fusion's TwiML reply and opens the stream it names, with the token from it
3. sends your microphone as 20 ms frames of 8 kHz μ-law, the way a phone line carries it
4. plays the agent back at phone quality, echoes each `mark` once that audio has played, and
   drops what hasn't played when fusion sends `clear` (you interrupted)

What you hear is what a caller would hear. The phone network itself is the only thing missing;
test that on a real Twilio number before you rely on it.

The simulator isn't part of the `fusion-runtime` package. It's a tool for working on phone calls,
and fusion's tests use its Twilio side (`twilio_side.py`) to check the phone endpoints.

## Use it

In one terminal, start an agent with phone calls turned on. The token can be anything here, as
long as both sides use the same one:

```bash
export FUSION_TWILIO_AUTH_TOKEN=local-test-token
frun up examples/tools_agent.py
```

In another, call it:

```bash
export FUSION_TWILIO_AUTH_TOKEN=local-test-token
python3 tools/phone_sim/phone_sim.py
```

Talk, and interrupt the agent whenever you like. Ctrl+C hangs up. Echo cancellation is on, so
the laptop's speakers work; `--no-aec` turns it off for headphones.

Without a microphone, say a recording and keep what the agent said:

```bash
python3 tools/phone_sim/phone_sim.py --say question.wav --record agent.wav
```

| Option | |
|---|---|
| `--url` | the fusion server (default `http://127.0.0.1:8000`) |
| `--say WAV` | play this recording as the caller instead of the microphone (any rate, mono or stereo) |
| `--listen S` | with `--say`, how long to wait for the answer (15 s) |
| `--record WAV` | save the agent's side of the call, at 8 kHz, as the caller heard it |
| `--caller` | the caller's number Twilio would send (`+15550100001`) |
| `--auth-token` | instead of `FUSION_TWILIO_AUTH_TOKEN` |
| `--no-aec` | no echo cancellation |

Needs `pip install "fusion-runtime[talk]"` for the microphone and speaker.

## On a real number

When you have a Twilio number, point its voice webhook ("A call comes in") at
`https://<your server>/v1/phone/twilio`, and set `FUSION_TWILIO_AUTH_TOKEN` to the account's real
auth token where fusion runs. Behind a proxy or on a pod, also set `FUSION_PUBLIC_URL` to the address
Twilio calls (on a Runpod pod `frun up` sets it for you): Twilio signs that address, and fusion
checks the signature against it.
