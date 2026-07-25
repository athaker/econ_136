# Kalshi Market-Making Demo — Agents for Finance (Ai4 2026)

Companion code for the **"Agents for Finance"** talk by Avi Thaker and Daniel Sonner
(Ai4 2026, The Venetian, Las Vegas).

---

## ⚠️ FOR EDUCATIONAL PURPOSES ONLY

**This code exists to teach concepts on a conference stage. It is NOT a trading
system, and you should not trade with it.**

- **This is not financial advice.** Nothing here is a recommendation to buy or
  sell any contract, on Kalshi or anywhere else.
- **The agent is not trained.** The neural network is randomly initialized and
  never learns — its decisions are effectively random. It is a teaching prop
  that demonstrates the *structure* of a market-making agent (state → action →
  reward), not a strategy.
- **You will most likely lose money if you run this against real markets.**
  It has no edge, pays fees when it flattens, and gets adversely selected.
- **Do not run this with money you are not fully prepared to lose.** Better:
  don't run it with real money at all — use Kalshi's demo environment (see below).
- **Event contracts are regulated products.** Trading on a CFTC-regulated
  exchange is subject to Kalshi's terms of service and applicable law. You are
  responsible for your own eligibility and compliance.
- **No warranty.** This software is provided "as is," without warranty of any
  kind. The authors accept no liability for any losses or damages arising from
  its use.

If you take one thing from this repo, let it be the ideas — not the P&L.

---

## What this is

A minimal, readable demonstration of an agent that *makes* markets on a Kalshi
event contract:

1. A tiny Q-network maps market state (best YES bid/ask, book depth, inventory)
   to one of three actions: `HOLD`, `BID`, or `ASK`.
2. `BID` / `ASK` post **passive limit orders one tick inside the touch** — the
   agent provides liquidity rather than taking it.
3. Fills are tracked, P&L is marked to mid during each cycle, and inventory is
   force-flattened (crossing the spread, paying taker fees) at cycle end.

The point of the demo: prices on a prediction market are probabilities, spreads
are the market maker's compensation, and adverse selection + inventory risk are
what that compensation pays for.

## Setup

1. Create a Kalshi account and generate an API key:
   **Settings → API** — you'll receive a key ID and download an RSA private key
   (`.pem`).
2. Export credentials:

   ```bash
   export KALSHI_ACCESS_KEY="your-key-id"
   export KALSHI_PRIVATE_KEY_PATH="/path/to/kalshi_private_key.pem"
   ```

3. Install dependencies:

   ```bash
   pip install requests cryptography torch
   ```

## Strongly recommended: rehearse without real money

- **Demo environment.** Kalshi provides a paper-trading environment with
  separate credentials. In the script, switch `BASE_URL` to the demo host
  (noted in the config section of the code) and use demo-account keys.
- **Dry run.** Set `DRY_RUN = True` to print orders instead of sending them.

## Configuration

All knobs are at the top of `kalshi-market-making-demo.py`:

| Setting | Default | Meaning |
|---|---|---|
| `TICKER` | `CONTROLH-2026-D` | Market to quote (2026 House control) |
| `ORDER_SIZE` | 10 | Contracts per quote (max loss ≤ $0.99 each) |
| `MAX_INVENTORY` | 30 | Absolute net position cap |
| `CYCLES` | 25 | Quoting cycles per session |
| `SECONDS_PER_CYCLE` | 45 | Length of each cycle |
| `DRY_RUN` | False | Print orders instead of sending |

To browse open markets by keyword:

```bash
python kalshi-market-making-demo.py --list house
```

## Safety behavior built into the script

- All resting orders are canceled on exit — including `Ctrl-C` — via
  `atexit` and a `finally` block.
- Inventory is capped and force-flattened at the end of every cycle.
- Quotes are always passive (never cross the spread) during the cycle.

These reduce the chance of orphaned orders during a live demo. They do **not**
make the strategy safe or profitable.

## Known limitations (also: the curriculum)

- **Untrained network** — add a real reward signal, experience replay, and
  training if you want the "learning" in Q-learning. That's the exercise.
- **1-second REST polling** — real market makers consume the WebSocket feed and
  re-quote on book deltas, not on a timer.
- **No fee model in quoting decisions**, no queue-position awareness, no news
  awareness. Probabilities jump on news faster than a polling loop can react.
- **Thick books** — on very liquid markets the spread is often 1¢, so "one tick
  inside" becomes joining the back of the queue, and fills may be rare.

Each of these is a good starting point for your own (paper-traded!) experiments.

## Disclaimer, one more time

Educational purposes only. Not investment advice. Not a trading system. The
authors and their affiliates are not responsible for any trading losses,
account restrictions, or other consequences of running this code. **Please
don't try this at home — and definitely not with size.**
