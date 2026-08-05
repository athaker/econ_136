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
   **Settings → API** — you get **two** things: a **key ID** (a UUID, shown
   on screen once) and a downloaded **RSA private key** file. You need both;
   the private key alone will not authenticate.
2. Save the credentials. Simplest is two files in the repo root — both are
   gitignored:

   ```bash
   # key.txt     <- the downloaded RSA private key (already in place)
   echo "1a2b3c4d-5e6f-7890-abcd-ef1234567890" > key_id.txt   # your key ID
   ```

   Environment variables still win if set:

   ```bash
   export KALSHI_ACCESS_KEY="your-key-id"
   export KALSHI_PRIVATE_KEY_PATH="./key.txt"
   ```

   The key ID **cannot be derived from the private key** — it is a server-side
   identifier, and the `.pem` contains nothing but the RSA key. It is not a
   secret, though (it rides in a plaintext header on every request), so Kalshi
   lists it in Settings → API next to each key. Only the private key was
   download-once.

   Using `key_id.txt` is the more robust option: a placeholder left in a shell's
   `KALSHI_ACCESS_KEY` will silently override every later command in that
   terminal, which produces a confusing `INVALID_PARAMETER` 401. The loader
   ignores known placeholder values for exactly this reason.

3. Install dependencies:

   ```bash
   pip install requests cryptography torch
   ```

4. Verify auth before anything else — this places no orders:

   ```bash
   python check_auth.py
   ```

   It checks your env vars, the key file, and authenticated calls against both
   the production and demo hosts, then tells you what to fix. Kalshi's two 401s
   mean different things and it decodes them for you: `INVALID_PARAMETER` means
   the key ID is malformed (usually a leftover placeholder);
   `token_authentication_failure` means the ID is well-formed but the
   credential did not validate (wrong/revoked ID, or `key.txt` is not its
   matching private key).

### A note on the API version

This code targets the **post-March-2026** Kalshi API. The Q1 2026 migration
removed every legacy integer field, so older examples (including v2 of this
demo) fail with `KeyError: 'orderbook'`:

| Was | Is now |
|---|---|
| `orderbook.yes` / `.no` | `orderbook_fp.yes_dollars` / `.no_dollars` |
| `yes_bid` (int cents) | `yes_bid_dollars` (`"0.8300"`) |
| `count` (int) | `count_fp` (`"1.00"`, may be fractional) |
| `tick_size` | `price_ranges[].step` — **tapered**: 0.1¢ below 10¢ and above 90¢, 1¢ between |
| `POST /portfolio/orders` with `action`+`side` | `POST /portfolio/events/orders` with `side: bid\|ask` |

`kalshi_client.py` wraps all of this. It works internally in **mils**
(thousandths of a dollar = deci-cents) so price math stays exact integer math.

## The live order book viewer

`book_viewer.py` renders the book as a text depth ladder — crowd bids below,
crowd asks above, price down the middle — with a marker for **where you are in
the book**: your resting quotes (magenta `◀ YOU 10`), your inventory, and the
microprice you're quoting around.

Run it **standalone** to watch a market. It places no orders — the safest way
to rehearse and to warm up an audience. Kalshi's market data is public, so the
ladder and `--scan` work **before your API key is sorted out**; only the
inventory and balance lines need credentials:

```bash
python book_viewer.py --scan                   # find something liquid
python book_viewer.py KXZEC15M-26AUG022030-30  # watch it
python book_viewer.py TICKER --depth 12

export KALSHI_ACCESS_KEY="your-key-id"         # adds inventory + balance
export KALSHI_PRIVATE_KEY_PATH="./key.txt"
```

```
 CONTROLH-2026-D  Which party wins the House?
 bid 45¢x610  ask 46¢x95  spread 1¢  mid 45.5¢  micro 45.38¢  imb +73%
──────────────────────────────────────────────────────────────────────
                    CROWD BIDS   PRICE   CROWD ASKS
                              │ 48¢ │  ██████████████████████ 2,200
                              │ 47¢ │  ████████ 700 YOU 10 ▶
                              │ 46¢ │  █ 95
                    ╌╌╌╌╌╌╌╌◆╌╌╌╌╌╌╌╌ fair 45.38¢  (1¢ wide)
        ◀ YOU 10 610 ███████  │ 45¢ │
    1,500 █████████████████▌  │ 44¢ │
──────────────────────────────────────────────────────────────────────
 INVENTORY +12  QUOTE 45¢/47¢ (2¢ wide)  P&L $-0.42  POSTURE LEAN_SELL
```

Reading it: bar length is resting size at that price, so you can see the wall
the crowd has built and how far a taker would have to walk to clear it. `imb`
is top-of-book size imbalance — the pressure the microprice leans on. The `◆`
sits where fair value falls *within* the spread.

During the trading demo the same ladder replaces the scrolling status line, and
fills scroll in a small log beneath it. Toggle with `SHOW_LADDER` (and
`LADDER_DEPTH`, `EVENT_LINES`) at the top of the demo script.

## Strongly recommended: rehearse without real money

- **Demo environment.** Kalshi provides a paper-trading environment with
  separate credentials. In the script, switch `BASE_URL` to the demo host
  (noted in the config section of the code) and use demo-account keys.
- **Dry run.** Set `DRY_RUN = True` to print orders instead of sending them.

## Finding a market

Don't guess. `--scan` ranks markets that are **actually trading** by the edge a
maker can capture net of fees:

```bash
python kalshi-market-making-demo.py --scan
```

It reads the exchange-wide recent-trade feed rather than `/markets?status=open`,
because that listing is dominated by tens of thousands of zero-volume parlay
(`KXMVE*`) markets — a naive scan returns nothing tradeable.

What to pick: **thin `BIDSZ`/`ASKSZ`** (a short queue means you get filled
instead of sitting behind 200k contracts) and a **positive `NET/1`**.

### Category focus

The talk is about crowds forecasting *events*, so the scan is biased toward
elections, politics and culture, and **excludes Sports and Crypto** by default.
Preferred categories sort to the top and print in cyan:

> Elections · Politics · Mentions · Entertainment · Social · World · Companies ·
> Science and Technology

```bash
python book_viewer.py --scan                           # default focus
python book_viewer.py --scan --focus                   # the wide on-theme families
python book_viewer.py --scan --only Elections,Politics  # narrow hard
python book_viewer.py --scan --series KXTRUMPSAY        # one family
python book_viewer.py --scan --sports                   # put sports back
```

### Will it actually trade? Read these two columns

| Column | Meaning |
|---|---|
| **`NET/1`** | Half-spread captured as maker, minus the taker fee to flatten 1 contract. **Is there edge?** |
| **`V24H`** | Contracts traded in the last 24h. **Does anyone trade this?** |
| **`FILL/S`** | Expected contracts *you* trade in one 15-min session, given the queue ahead of you. Green (`>2`) means a fill is likely. |

You need `NET/1 > 0` **and** `FILL/S` green. They pull against each other, and
seeing that is the point:

```
KXWTI-26AUG0514-T74.99    1c wide   V24H 64,342   NET -0.5c   FILL/S 111.7  <- trades constantly, no edge
KXWMTFT-26AUG08-T1220    30c wide   V24H    216   NET +13.0c  FILL/S   0.8  <- great edge, nobody trades
KXRAIN-26AUG05-NOLA       7c wide   V24H  5,029   NET  +1.5c  FILL/S   3.3  <- the one worth quoting
```

Use `--min-vol 500` to drop markets that don't trade enough to matter.

**Why `V24H` and not a trade count:** the exchange-wide trade feed this scanner
reads spans only about **40 seconds**. A market that prints 30×/day shows *zero*
trades in that window. An earlier version of this tool scored on that count and
systematically buried exactly the markets worth quoting.

`--focus` scans the **Mentions** families (`KXTRUMPMENTION`, `KXTRUMPSAY`,
`KXMAMDANIMENTION`, `KXFOXNEWSMENTION`) by enumerating their series directly.
The recent-trade feed under-ranks these: a market printing 30×/day rarely
appears in a two-minute window, but it is exactly what you want to quote.

Categories come from `GET /series/{ticker}`, cached in `.category_cache.json`.
Two traps worth knowing if you change this: `/events?category=X` is silently
ignored by the API (it returns everything), and the bulk `/events` listing does
not cover every series — filtering that way let tennis, UFC and soccer through
as `Unknown`. Only the per-series lookup is authoritative.

### The tension with election markets

Elections tell the better story but make the worse market-making demo. The
headline contracts are penny-wide and enormously deep — `KXMAYORLA-26-KBAS`
quotes 62¢/63¢ with **145,000 contracts** at the bid. You will never reach the
front of that queue in a 20-minute slot, and `NET/1` is correctly negative.

Look further down the same category for the thin ones: a foreign by-election or
a nomination market quoting 8¢/17¢ with a couple hundred contracts a side gives
you a real spread and a reachable queue. That contrast is itself worth a slide.

## Configuration

All knobs are at the top of `kalshi-market-making-demo.py`:

| Setting | Default | Meaning |
|---|---|---|
| `ORDER_SIZE` | 1 | Contracts per quote |
| `MAX_INVENTORY` | 1 | Absolute net position cap |
| `MIN_EDGE_C` | 0.5 | Cents of profit required per side, after fees |
| `MAX_IMPROVE_C` | 1.0 | Most we may improve on the touch — the queue-jump cap |
| `FAIR_TRUST` | 0.6 | How much of the microprice to believe vs. the plain mid |
| `SKEW_PER_CONTRACT_C` | 0.5 | Fair-value shift per contract of inventory |
| `BALANCE_RESERVE` | 0.25 | Dollars kept unencumbered |
| `STARTER_CROSS` | False | v2 crossed the spread for a guaranteed opening fill |
| `DRY_RUN` | False | Print orders instead of sending |

### Running several markets at once

Run one process per market, **each in its own terminal window** — the ladder
repaints with a screen-clear, so two in the same pane will fight over it.

Give each instance its own slice of the account with `--budget`:

```bash
python kalshi-market-making-demo.py TICKER_A --budget 1.00   # terminal 1
python kalshi-market-making-demo.py TICKER_B --budget 1.00   # terminal 2
```

**This is required, not optional.** Every instance reads the same account
balance and, without `--budget`, sizes as if it owned all of it — so two
instances on $2.40 each conclude they can post two lots, and whichever gets
there second has its orders rejected for insufficient collateral. `--budget`
caps how much *this* instance will tie up; the real balance still binds on top
of it, so an absurd `--budget 99` cannot overdraw you.

One `python flatten.py` at the end cleans up after all of them — it sweeps
every ticker, not just one.

### Sizing for a tiny account

A resting bid for 1 YES at price *P* locks *P* dollars; a resting ask for 1 YES
is economically a NO bid at *(1 − P)*, locking *(1 − P)*. So **a one-lot
two-sided quote ties up about $1.00 of collateral regardless of price.** On a
$2.50 account that is roughly two lots, which is why `ORDER_SIZE` and
`MAX_INVENTORY` both default to 1. The script re-checks your balance as it runs
and pauses quoting rather than firing orders that will be rejected.

### How the quote is built

- Fair value is the microprice, **shrunk toward the plain mid** by `FAIR_TRUST`.
  On a thin lopsided book the raw microprice lurches toward whichever side has
  less size, and will happily imply a "fair" value 15¢ from the crowd's bid.
- Total width is pinned at the point where a round trip still clears the taker
  fee to flatten — `MIN_EDGE_C` on top. Every order is `post_only`, so we are
  always the **maker** and pay no entry fee.
- A lean does **not** tighten the quote, it *slides* it: the side we want filled
  moves closer to fair, the other side moves out by the same amount. (Leaning
  toward the touch would be backwards on a wide book — there the best bid sits
  far below fair, so "joining the bid" is the least fillable price available.)
- `MAX_IMPROVE_C` caps how far we may jump the queue, so a lurching fair value
  can never make us the best bid by a mile and get us run over.

## Panic button

The in-session kill switch only knows about the ticker it was quoting. If a run
died uncleanly, or you just want to prove the account is flat:

```bash
python flatten.py        # dry run — lists resting orders and open positions
python flatten.py --go   # cancel everything, close everything, all tickers
```

## Two P&L numbers

The footer shows both, and the difference is the lesson:

- **`mark`** values inventory at the microprice — what it's theoretically worth.
- **`liq`** is what you'd actually net closing right now, crossing the spread
  and paying the taker fee.

Buying 1 contract at the ask on a 12¢-wide book marks at −$0.06 but liquidates
at −$0.14. That 8¢ gap is inventory risk, and the spread is what compensates a
market maker for carrying it.

## Safety behavior built into the script

- All resting orders are canceled on exit — including `Ctrl-C` — via `atexit`.
- Inventory is capped and force-flattened at the end, with escalation: the
  flattening order walks through the book until the position verifies flat, and
  shouts loudly if it does not.
- Quotes are always `post_only`, so they can never cross the spread.
- `MAX_IMPROVE_C` bounds how aggressive a quote can be relative to the touch.
- Quoting pauses instead of firing orders your balance cannot collateralize.

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
