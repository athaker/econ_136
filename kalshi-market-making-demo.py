"""
Kalshi Market-Making Demo v3 — Agents for Finance (2026)
Avi Thaker & Daniel Sonner

A small Q-learning agent that MAKES markets on a Kalshi contract, sized for a
tiny account and rendered as a live depth ladder (see book_viewer.py).

Quoting logic (v3):
  * Fair value = microprice (size-weighted mid of best bid/ask)
  * Fair is skewed by inventory (long -> shade down, short -> shade up)
  * Quotes are ALWAYS two-sided, and the width is floored at the level where a
    round trip still pays after the taker fee to flatten (MIN_EDGE_C)
  * The agent picks a lean (SYMMETRIC / LEAN_BUY / LEAN_SELL); the leaned side
    steps toward the touch to actually get filled, and the other side is pushed
    out to preserve the required width
  * All orders are post_only -> we are always the MAKER, so entry fees are zero
  * ALL positions are force-flattened at the end, with escalation

v3 targets the post-March-2026 API (dollar strings, fixed-point counts,
tapered deci-cent ticks, /portfolio/events/orders). See kalshi_client.py.

DEMO ONLY. NOT FINANCIAL ADVICE. Please don't try this at home.

Setup:
  export KALSHI_ACCESS_KEY="your-key-id"
  export KALSHI_PRIVATE_KEY_PATH="./key.txt"
  pip install requests cryptography torch

Usage:
  python kalshi-market-making-demo.py --scan            # find a live market
  python kalshi-market-making-demo.py TICKER            # quote it
  python kalshi-market-making-demo.py TICKER --dry-run  # print, don't send
"""

import atexit
import os
import random
import sys
import time

import requests
import torch
import torch.nn as nn

import book_viewer
import kalshi_client as kc

# --- Configuration -----------------------------------------------------------

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
# Rehearsal without real money: https://demo-api.kalshi.co/trade-api/v2

# No default ticker on purpose. Any hard-coded market goes stale — the one that
# used to live here was a 15-minute crypto contract that expires the same hour.
# Run --scan and pass the ticker explicitly.

RUN_MINUTES = 15
ORDER_SIZE = 1              # contracts per quote — $2.50 buys about two
MAX_INVENTORY = 1           # absolute net YES cap
MIN_EDGE_C = 0.5            # cents of profit required per side, after fees
SKEW_PER_CONTRACT_C = 0.5   # cents of fair-value shift per contract held

# Safety rails. On a thin, lopsided book the microprice lurches toward whichever
# side has less size, which on a wide market can imply a "fair" value 15c away
# from the crowd's bid. Two guards:
MAX_IMPROVE_C = 1.0         # we may improve on the touch by at most this much,
                            # so we can never become the best bid by a mile
FAIR_TRUST = 0.6            # shrink the microprice toward the plain mid; the
                            # rest is book noise, not information

BALANCE_RESERVE = 0.25      # dollars kept unencumbered as a buffer

# Running several markets at once: each instance reads the SAME account balance
# and would size as if it owned all of it, so two instances on $2.40 both think
# they can post two lots and the second one's orders get rejected. Give each
# instance its own slice with --budget, e.g.
#     python kalshi-market-making-demo.py TICKER_A --budget 1.00
#     python kalshi-market-making-demo.py TICKER_B --budget 1.00
# Each still refuses to exceed the real balance; this only caps it further.
MAX_COLLATERAL = None       # dollars this instance may tie up (None = all)

# v2 had STARTER_CROSS, which crossed the spread once at launch to guarantee a
# visible fill on stage. It is gone rather than disabled: it is a deliberate
# taker loss every run, and on a small account that is real money. The lean
# earns a fill instead. Removed, not left as a dead switch.

DRY_RUN = False
SHOW_BALANCE = False        # sessions get recorded and posted — keep the
                            # account balance off the projector
SHOW_LADDER = True
LADDER_DEPTH = 6
EVENT_LINES = 6

KEY_ID = kc.load_key_id()             # $KALSHI_ACCESS_KEY, else key_id.txt
PRIVATE_KEY_PATH = kc.load_key_path()

EVENTS = []


def event(msg):
    EVENTS.append(f"{time.strftime('%H:%M:%S')}  {msg}")
    del EVENTS[:-EVENT_LINES]
    if not SHOW_LADDER:
        print(msg)


# --- 1. Q-Learning Agent (same brain as v2) ----------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    """Epsilon-greedy agent choosing a quoting POSTURE, not a trade.

    NOTE: this network is randomly initialized and never trained. It is a
    teaching prop for the state -> action -> reward structure, not a strategy.
    """

    def __init__(self, state_size, epsilon=0.2):
        self.q_network = QNetwork(state_size, 3)
        self.actions_map = {0: "SYMMETRIC", 1: "LEAN_BUY", 2: "LEAN_SELL"}
        self.epsilon = epsilon
        self.last_reason = ""

    def choose_action(self, state, inventory):
        valid = ["SYMMETRIC"]
        if inventory < MAX_INVENTORY:
            valid.append("LEAN_BUY")
        if inventory > -MAX_INVENTORY:
            valid.append("LEAN_SELL")

        if random.random() < self.epsilon:
            action = random.choice(valid)
            self.last_reason = f"explore -> {action}"
            return action
        with torch.no_grad():
            q = self.q_network(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
            action = self.actions_map[torch.argmax(q).item()]
            if action not in valid:
                action = "SYMMETRIC"
            self.last_reason = f"exploit -> {action}"
            return action


# --- 2. Quoting --------------------------------------------------------------

def required_half_width(fair_mils):
    """Half-width (mils) that keeps a round trip profitable.

    We are always post_only, so ENTRY costs no fee. The cost we must cover is
    the taker fee to flatten whatever we end up holding, plus the edge we
    actually want to earn.
    """
    exit_fee_mils = kc.taker_fee_dollars(1, fair_mils) * 1000
    return exit_fee_mils / 2.0 + MIN_EDGE_C * 10


def make_quotes(market, bid, ask, inventory, posture, fair):
    """Two-sided quotes around an inventory-skewed fair value, in mils.

    The TOTAL width is pinned at 2 * required_half_width, so every round trip
    clears fees no matter the posture. A lean does not tighten the quote — it
    SLIDES it, so the side we want filled sits closer to fair (and therefore
    closer to the crowd) while the other side is pushed out by the same amount.

    Leaning toward the touch would be wrong on a wide book: there the best bid
    can sit far below fair, so "joining the bid" is the least fillable price
    available. Sliding toward fair is what actually earns a fill.
    """
    tick = kc.tick_mils(market, int(fair))
    half = max(required_half_width(fair), tick / 2.0)
    edge = MIN_EDGE_C * 10                      # minimum edge we always keep
    slide = max(0.0, half - edge)               # how far a lean may shift us

    center = fair - SKEW_PER_CONTRACT_C * 10 * inventory
    if posture == "LEAN_BUY":
        center += slide                          # raises the bid toward fair
    elif posture == "LEAN_SELL":
        center -= slide                          # lowers the ask toward fair

    qbid = kc.round_to_tick(market, int(round(center - half)), -1)
    qask = kc.round_to_tick(market, int(round(center + half)), +1)

    # Aggression cap: never jump the queue by more than MAX_IMPROVE_C. Without
    # this, a lurching microprice on a wide book will happily post a bid far
    # above the crowd's best bid and get adversely selected instantly.
    qbid = min(qbid, kc.round_to_tick(market, bid + int(MAX_IMPROVE_C * 10), -1))
    qask = max(qask, kc.round_to_tick(market, ask - int(MAX_IMPROVE_C * 10), +1))

    # stay passive: never cross the book (post_only would reject us anyway)
    qbid = min(qbid, kc.round_to_tick(market, ask - tick, -1))
    qask = max(qask, kc.round_to_tick(market, bid + tick, +1))

    qbid = max(tick, qbid)
    qask = min(1000 - tick, qask)
    if qask - qbid < tick:
        qask = qbid + tick
    return int(qbid), int(qask)


def liquidation_value(inventory, bid, ask):
    """Dollars we'd actually net by closing RIGHT NOW, crossing the spread.

    Marking inventory to the mid flatters the P&L: it assumes you can exit at a
    price nobody is offering. The gap between the mark and this number is the
    inventory risk a market maker's spread is compensation for — which is the
    whole point of the talk, so put it on the screen.
    """
    if abs(inventory) < 0.005:
        return 0.0
    if inventory > 0:                       # sell into the bid
        return inventory * bid / 1000.0 - kc.taker_fee_dollars(inventory, bid)
    return inventory * ask / 1000.0 - kc.taker_fee_dollars(-inventory, ask)


def affordable_size(qbid, qask, balance):
    """How many contracts we can post two-sided given the cash on hand.

    A resting bid for 1 YES at P locks P dollars. A resting ask for 1 YES is
    economically a NO bid at (1 - P), locking (1 - P). So a one-lot two-sided
    quote ties up roughly $1.00 regardless of price.
    """
    per_lot = qbid / 1000.0 + (1000 - qask) / 1000.0
    usable = max(0.0, balance - BALANCE_RESERVE)
    if MAX_COLLATERAL is not None:
        usable = min(usable, MAX_COLLATERAL)   # our slice when running several
    return int(usable // per_lot) if per_lot > 0 else 0


# --- 3. Session --------------------------------------------------------------

def run_session(client, agent, ticker, market):
    title = market.get("title", "")
    try:
        inventory = client.get_position(ticker)
    except requests.RequestException as e:
        event(f"could not read position ({e}); assuming flat")
        inventory = 0.0
    if inventory:
        event(f"starting with existing inventory of {inventory:+g}")

    try:
        balance = client.get_balance_dollars()
    except requests.RequestException as e:
        event(f"could not read balance ({e}); assuming $0")
        balance = 0.0

    cash = 0.0                 # realized cash flow, dollars
    fees = 0.0                 # actual fees from fills, dollars
    seen = set()
    since_ts = int(time.time())
    resting = {"bid": (None, None), "ask": (None, None)}   # side -> (id, price)
    n_fills = 0

    def process_fills():
        nonlocal cash, fees, n_fills
        try:
            fills = client.get_fills(ticker, since_ts)
        except requests.RequestException:
            return
        for f in fills:
            fid = f.get("fill_id") or f.get("trade_id")
            if not fid or fid in seen:
                continue
            seen.add(fid)
            qty = float(f.get("count_fp", 0))
            px = kc.mils(f.get("yes_price_dollars", "0"))
            fees += float(f.get("fee_cost", 0) or 0)
            n_fills += 1
            # book_side tells us which side of the book WE were on
            side = (f.get("book_side") or f.get("action") or "").lower()
            bought = side in ("bid", "buy")
            if bought:
                cash -= qty * px / 1000.0
                event(f"\033[92mFILL  bought {qty:g} YES @ {kc.cents(px)}\033[0m")
            else:
                cash += qty * px / 1000.0
                event(f"\033[91mFILL  sold   {qty:g} YES @ {kc.cents(px)}\033[0m")

    def kill_all(tag="SESSION END"):
        """Cancel everything, then flatten with escalating aggression."""
        nonlocal inventory
        print(f"\n{tag}: canceling quotes and flattening...")
        try:
            client.cancel_all(ticker)
        except Exception as e:
            print(f"  warn: cancel_all failed ({e})")
        if DRY_RUN:
            return True
        for attempt in range(10):
            process_fills()
            try:
                inventory = client.get_position(ticker)
            except requests.RequestException:
                time.sleep(1)
                continue
            if abs(inventory) < 0.005:
                print("POSITION VERIFIED FLAT.")
                return True
            try:
                b, a, _, _ = kc.touch(*kc.normalize(client.get_orderbook(ticker)))
                tick = kc.tick_mils(market, b)
                if inventory > 0:            # long -> hit the bid, walking down
                    px = kc.round_to_tick(market, b - attempt * tick, -1)
                    client.place_order(ticker, "ask", abs(inventory), px,
                                       post_only=False)
                else:                        # short -> lift the ask, walking up
                    px = kc.round_to_tick(market, a + attempt * tick, +1)
                    client.place_order(ticker, "bid", abs(inventory), px,
                                       post_only=False)
                print(f"  flatten attempt {attempt + 1}: {abs(inventory):g} @ "
                      f"{kc.cents(px)}")
            except requests.RequestException as e:
                print(f"  flatten attempt {attempt + 1} rejected: {e}")
            time.sleep(1.5)
            try:
                client.cancel_all(ticker)
            except Exception:
                pass
        process_fills()
        try:
            inventory = client.get_position(ticker)
        except requests.RequestException:
            inventory = None
        if inventory:
            print(f"CRITICAL: still holding {inventory:+g}. Close it manually "
                  f"in the Kalshi UI NOW.")
            return False
        print("POSITION VERIFIED FLAT.")
        return True

    atexit.register(lambda: kill_all("EXIT SAFETY"))

    deadline = time.time() + RUN_MINUTES * 60
    cycle = 0
    size = ORDER_SIZE
    print(f"\nQuoting {ticker} for {RUN_MINUTES} minutes...")

    while time.time() < deadline:
        time.sleep(1)
        cycle += 1
        try:
            book = client.get_orderbook(ticker, depth=LADDER_DEPTH + 6)
        except requests.RequestException as e:
            event(f"waiting for market data ({e})")
            continue

        bids, asks = kc.normalize(book)
        bid, ask, bid_sz, ask_sz = kc.touch(bids, asks)

        process_fills()
        try:
            inventory = client.get_position(ticker)
        except requests.RequestException:
            pass

        micro = kc.microprice(bid, ask, bid_sz, ask_sz)
        fair = (bid + ask) / 2 + FAIR_TRUST * (micro - (bid + ask) / 2)

        # Reasons we may be alive but not quoting. The ladder is drawn either
        # way — a frozen screen with no explanation is the worst outcome.
        banner = None
        if not bids or not asks:
            banner = "NOT QUOTING — book is empty"
            if cycle % 15 == 1:                # re-check whether it even trades
                try:
                    st = client.get_market(ticker).get("status")
                    if st != "active":
                        banner = f"MARKET IS {str(st).upper()} — nothing to quote"
                        event(f"market status is {st}; stopping")
                        break
                except requests.RequestException:
                    pass
        state = [bid / 1000, ask / 1000, min(bid_sz, 1e4) / 1e4,
                 min(ask_sz, 1e4) / 1e4, fair / 1000,
                 inventory / max(1, MAX_INVENTORY)]
        posture = agent.choose_action(state, inventory)
        qbid, qask = make_quotes(market, bid, ask, inventory, posture, fair)

        # size to the cash actually available
        can = affordable_size(qbid, qask, balance)
        size = max(0, min(ORDER_SIZE, can))
        if size == 0 and not banner:
            banner = (f"NOT QUOTING — balance ${balance:.2f} cannot collateralize "
                      f"a two-sided lot (needs about $1.00)")

        mark = cash + inventory * fair / 1000.0 - fees
        liq = cash + liquidation_value(inventory, bid, ask) - fees

        if SHOW_LADDER:
            book_viewer.draw(
                book, ticker=ticker, title=title, depth=LADDER_DEPTH,
                fair=fair, step=kc.tick_mils(market, int(fair)),
                my_bid=resting["bid"][1], my_ask=resting["ask"][1],
                my_bid_size=size, my_ask_size=size,
                inventory=inventory, pnl=mark, pnl_liq=liq, posture=posture,
                balance=balance if SHOW_BALANCE else None,
                events=EVENTS, banner=banner,
                note=f"cycle {cycle} · fills {n_fills} · fees ${fees:.2f} · "
                     f"{agent.last_reason} · {int(deadline - time.time())}s left"
                     f" · Ctrl-C = flatten")
        else:
            print(f"| {kc.cents(bid)} x {kc.cents(ask)} | inv {inventory:+g} | "
                  f"quote {kc.cents(qbid)}/{kc.cents(qask)} | "
                  f"pnl ${mark:+.2f} | {posture}"
                  + (f" | {banner}" if banner else ""))

        if DRY_RUN or size == 0 or banner:
            continue

        # refresh a side only when its target price moved
        for side, target, cap_ok in (
                ("bid", qbid, inventory < MAX_INVENTORY),
                ("ask", qask, inventory > -MAX_INVENTORY)):
            if not cap_ok:
                if resting[side][0]:
                    client.cancel(resting[side][0])
                    resting[side] = (None, None)
                continue
            if resting[side][1] == target:
                continue
            if resting[side][0]:
                client.cancel(resting[side][0])
            try:
                oid = client.place_order(ticker, side, size, target,
                                         post_only=True)
                resting[side] = (oid, target)
            except requests.RequestException as e:
                resting[side] = (None, None)
                event(f"{side} rejected @ {kc.cents(target)}: {str(e)[:70]}")

        if cycle % 20 == 0:
            try:
                balance = client.get_balance_dollars()
            except requests.RequestException:
                pass

    ok = kill_all()
    process_fills()
    realized = cash - fees
    print("\n" + "-" * 70)
    print(f"Demo finished. Fills: {n_fills} | Fees paid: ${fees:.2f}")
    print(f"Final realized P&L: ${realized:+.2f}"
          + ("" if ok else "   (WARNING: position not flat!)"))


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--dry-run" in argv:
        DRY_RUN = True
        argv.remove("--dry-run")
    if "--budget" in argv:
        i = argv.index("--budget")
        MAX_COLLATERAL = float(argv[i + 1])
        del argv[i:i + 2]

    if not KEY_ID:
        print("Set KALSHI_ACCESS_KEY (your key ID from Kalshi Settings -> API) "
              "and KALSHI_PRIVATE_KEY_PATH.")
        sys.exit(1)

    client = kc.KalshiClient(BASE_URL, KEY_ID, PRIVATE_KEY_PATH)

    if "--scan" in argv:
        only = series = None
        if "--only" in argv:
            only = [c.strip() for c in argv[argv.index("--only") + 1].split(",")]
        if "--focus" in argv:
            series = book_viewer.FOCUS_SERIES
        if "--series" in argv:
            series = [s.strip() for s in argv[argv.index("--series") + 1].split(",")]
        book_viewer.scan(client, include_sports="--sports" in argv, only=only,
                         series=series)
        sys.exit(0)

    positional = [a for a in argv if not a.startswith("--")]
    if not positional:
        print("Usage: python kalshi-market-making-demo.py TICKER [--dry-run]\n"
              "       python kalshi-market-making-demo.py TICKER --budget 1.00\n"
              "       python kalshi-market-making-demo.py --scan [--focus]\n\n"
              "No default ticker: any hard-coded market goes stale. Run --scan.")
        sys.exit(1)
    ticker = positional[0]

    market = client.get_market(ticker)
    print(f"Market: {market.get('title')}  [{ticker}]")
    print(f"Last: {market.get('last_price_dollars')} | "
          f"24h volume: {float(market.get('volume_24h_fp') or 0):,.0f} contracts")

    status = market.get("status")
    if status != "active":
        print(f"\nThis market is {str(status).upper()}, not active — it has "
              f"stopped trading and its book is gone.\nNothing can be quoted "
              f"here. Run --scan and pick a live ticker.")
        sys.exit(1)
    if not any(client.get_orderbook(ticker, depth=1).get(k)
               for k in ("yes_dollars", "no_dollars")):
        print(f"\nThis market is active but its book is completely empty.\n"
              f"Run --scan and pick one with real size on both sides.")
        sys.exit(1)
    try:
        bal = client.get_balance_dollars()
        print(f"Balance: ${bal:,.2f}")
        if bal < 1.0:
            print("WARNING: under $1.00 — a two-sided one-lot quote ties up "
                  "about $1.00 of collateral. Quoting will likely be paused.")
    except requests.RequestException as e:
        print(f"Could not read balance: {e}")

    agent = Agent(state_size=6, epsilon=0.2)
    print(f"Agent initialized with epsilon = {agent.epsilon} "
          f"(explores {agent.epsilon * 100:.0f}% of the time; network is "
          f"UNTRAINED).")

    if SHOW_LADDER:
        print("\033[?25l", end="", flush=True)
    try:
        run_session(client, agent, ticker, market)
    finally:
        if SHOW_LADDER:
            print("\033[?25h", end="", flush=True)
