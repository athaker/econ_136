"""
Kalshi Market-Making Demo v2 — Agents for Finance (2026)
Avi Thaker & Daniel Sonner

A small Q-learning agent that MAKES markets on a Kalshi election contract.
Quoting logic (v2):
  * Fair value = microprice (size-weighted mid of best bid/ask)
  * Quotes are skewed by inventory (long -> shade down, short -> shade up)
  * The agent chooses a "lean" (SYMMETRIC / LEAN_BUY / LEAN_SELL); the leaned
    side joins the touch so the demo actually trades on thick books
  * Optional starter cross at launch guarantees a visible fill on stage
  * ALL positions are killed (force-flattened) at the end, with escalation

DEMO ONLY. NOT FINANCIAL ADVICE. Please don't try this at home.

Setup:
  export KALSHI_ACCESS_KEY="your-key-id"
  export KALSHI_PRIVATE_KEY_PATH="/path/to/kalshi_private_key.pem"
  pip install requests cryptography torch

Usage:
  python kalshi-market-making-demo.py                # run the demo
  python kalshi-market-making-demo.py --scan senate  # rank live markets by
                                                     # spread x volume first

Prices are in CENTS (1-99). Contracts settle at 100c (YES) or 0c (NO).
"""

import atexit
import base64
import math
import os
import random
import sys
import time
import uuid

import requests
import torch
import torch.nn as nn
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# --- Configuration -----------------------------------------------------------

# Default: 2026 House control (Democratic side) — the most liquid election
# market, per the fallback plan: liquidity first, then lean one side to trade.
# Morning-of, run `--scan senate` and consider switching to a competitive
# individual race (e.g. the ~50/50 Ohio Senate market) for wider spreads and
# livelier passive fills.
TICKER = "CONTROLH-2026-D"

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
# Rehearsal without real money: https://demo-api.kalshi.co/trade-api/v2

RUN_MINUTES = 15            # 20-min slot INCLUDES Q&A and a hard stop:
                            # leave time for the kill sequence + wrap-up
REPORT_EVERY_S = 60         # cycle summary cadence
ORDER_SIZE = 10             # contracts per quote
STARTER_SIZE = 5            # size of the guaranteed opening fill
STARTER_CROSS = True        # cross the spread once at launch (guaranteed fill)
MAX_INVENTORY = 30          # absolute net YES cap

HALF_SPREAD = 1.0           # cents each side of fair value (when book allows)
SKEW_PER_CONTRACT = 0.05    # cents of fair-value shift per contract held
LEAN_EXTRA = 0.75           # cents of extra shade toward the leaned side

DRY_RUN = False             # True = print orders instead of sending
SHOW_BALANCE = False        # sessions are recorded & posted - keep the
                            # account balance off the projector

KEY_ID = os.environ.get("KALSHI_ACCESS_KEY", "")
PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")


# --- 1. Q-Learning Agent (same brain, new action space) ----------------------

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
    """Epsilon-greedy agent choosing a quoting POSTURE, not a trade."""

    def __init__(self, state_size, epsilon=0.2):
        self.q_network = QNetwork(state_size, 3)
        self.actions_map = {0: "SYMMETRIC", 1: "LEAN_BUY", 2: "LEAN_SELL"}
        self.epsilon = epsilon
        print(f"Agent initialized with epsilon = {epsilon}. "
              f"It will explore {epsilon * 100:.0f}% of the time.")

    def choose_action(self, state, inventory):
        valid = ["SYMMETRIC"]
        if inventory < MAX_INVENTORY:
            valid.append("LEAN_BUY")
        if inventory > -MAX_INVENTORY:
            valid.append("LEAN_SELL")

        if random.random() < self.epsilon:
            action = random.choice(valid)
            print(f"🎲 Exploring: posture {action}")
            return action
        with torch.no_grad():
            q = self.q_network(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
            action = self.actions_map[torch.argmax(q).item()]
            if action not in valid:
                action = "SYMMETRIC"
            print(f"🧠 Exploiting: posture {action}")
            return action


# --- 2. Minimal Kalshi REST client (RSA-PSS signed) --------------------------

class KalshiClient:
    def __init__(self, base_url, key_id, private_key_path):
        self.base_url = base_url
        self.key_id = key_id
        with open(private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)
        self.session = requests.Session()

    def _headers(self, method, path):
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method}{path}".encode()
        sig = self.private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "Content-Type": "application/json",
        }

    def _request(self, method, path, params=None, json=None):
        api_path = "/trade-api/v2" + path
        r = self.session.request(method, self.base_url + path, params=params,
                                 json=json, headers=self._headers(method, api_path),
                                 timeout=10)
        r.raise_for_status()
        return r.json() if r.text else {}

    # market data
    def get_market(self, ticker):
        return self._request("GET", f"/markets/{ticker}")["market"]

    def get_orderbook(self, ticker, depth=5):
        return self._request("GET", f"/markets/{ticker}/orderbook",
                             params={"depth": depth})["orderbook"]

    def list_open_markets(self, query, max_pages=10):
        out, cursor = [], None
        for _ in range(max_pages):
            params = {"status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            resp = self._request("GET", "/markets", params=params)
            for m in resp.get("markets", []):
                text = (m.get("title", "") + " " + m["ticker"]).lower()
                if query.lower() in text:
                    out.append(m)
            cursor = resp.get("cursor")
            if not cursor:
                break
        return sorted(out, key=lambda m: -(m.get("volume") or 0))

    # trading
    def place_limit(self, ticker, action, count, yes_price):
        if DRY_RUN:
            print(f"[DRY RUN] {action} {count} YES @ {yes_price}c")
            return None
        order = self._request("POST", "/portfolio/orders", json={
            "client_order_id": str(uuid.uuid4()),
            "ticker": ticker,
            "action": action,          # "buy" or "sell"
            "side": "yes",
            "type": "limit",
            "count": count,
            "yes_price": int(yes_price),
        })
        return order["order"]["order_id"]

    def cancel(self, order_id):
        if order_id is None:
            return
        try:
            self._request("DELETE", f"/portfolio/orders/{order_id}")
        except requests.HTTPError:
            pass  # already filled/canceled

    def cancel_all_resting(self, ticker):
        try:
            resp = self._request("GET", "/portfolio/orders",
                                 params={"ticker": ticker, "status": "resting"})
            for o in resp.get("orders", []):
                self.cancel(o["order_id"])
        except requests.HTTPError as e:
            print(f"warn: cancel_all failed ({e})")

    def get_fills(self, ticker, min_ts):
        resp = self._request("GET", "/portfolio/fills",
                             params={"ticker": ticker, "min_ts": min_ts, "limit": 200})
        return resp.get("fills", [])

    def get_position(self, ticker):
        resp = self._request("GET", "/portfolio/positions", params={"ticker": ticker})
        for p in resp.get("market_positions", []):
            if p["ticker"] == ticker:
                return p.get("position", 0)
        return 0

    def get_balance(self):
        return self._request("GET", "/portfolio/balance")["balance"]


# --- 3. Book math ------------------------------------------------------------

def best_prices(book):
    """(best_yes_bid, best_yes_ask, bid_size, ask_size) in cents.
    Kalshi books list resting YES bids and resting NO bids; a NO bid at q
    is a YES ask at 100 - q."""
    yes = book.get("yes") or []
    no = book.get("no") or []
    best_bid, bid_sz = (yes[-1][0], yes[-1][1]) if yes else (1, 0)
    best_ask, ask_sz = (100 - no[-1][0], no[-1][1]) if no else (99, 0)
    return best_bid, best_ask, bid_sz, ask_sz


def microprice(bid, ask, bid_sz, ask_sz):
    """Size-weighted mid: leans toward the heavier side's implied pressure."""
    if bid_sz + ask_sz == 0:
        return (bid + ask) / 2
    return (bid * ask_sz + ask * bid_sz) / (bid_sz + ask_sz)


def taker_fee_cents(count, price_cents):
    """General schedule: ceil(0.07 * C * P * (1-P)). Maker fee is 0 on most
    series (0.0175 formula on flagged ones) — verify your ticker's flag."""
    p = price_cents / 100.0
    return math.ceil(0.07 * count * p * (1 - p) * 100)


def make_quotes(bid, ask, bid_sz, ask_sz, inventory, posture):
    """Two-sided quotes around a skewed microprice. The leaned side joins the
    touch (so it can trade on a 1c-wide book); quotes never cross the spread."""
    fair = microprice(bid, ask, bid_sz, ask_sz)
    fair -= SKEW_PER_CONTRACT * inventory
    if posture == "LEAN_BUY":
        fair += LEAN_EXTRA
    elif posture == "LEAN_SELL":
        fair -= LEAN_EXTRA

    qbid = int(math.floor(fair - HALF_SPREAD))
    qask = int(math.ceil(fair + HALF_SPREAD))

    # leaned side steps up to the touch
    if posture == "LEAN_BUY":
        qbid = max(qbid, bid)
    if posture == "LEAN_SELL":
        qask = min(qask, ask)

    # never cross; keep inside [1, 99]; keep a valid two-sided quote
    qbid = max(1, min(qbid, ask - 1))
    qask = min(99, max(qask, bid + 1))
    if qbid >= qask:
        qbid = qask - 1
    return qbid, qask


# --- 4. Session --------------------------------------------------------------

def run_session(client, agent):
    inventory = client.get_position(TICKER)
    if inventory != 0:
        print(f"Note: starting with existing inventory of {inventory}.")
    cash_cents = 0.0
    fees_cents = 0.0
    seen_fills = set()
    since_ts = int(time.time())
    resting = {"buy": (None, None), "sell": (None, None)}  # side -> (id, px)
    n_fills = 0

    def process_new_fills():
        nonlocal inventory, cash_cents, n_fills
        try:
            fills = client.get_fills(TICKER, since_ts)
        except requests.HTTPError:
            return
        for f in fills:
            fid = f.get("trade_id") or f.get("fill_id") or \
                  f.get("order_id", "") + str(f.get("created_time"))
            if fid in seen_fills:
                continue
            seen_fills.add(fid)
            qty, px = f["count"], f["yes_price"]
            n_fills += 1
            if f["action"] == "buy":
                inventory += qty
                cash_cents -= qty * px
                print(f"✅ FILL: bought {qty} YES @ {px}c")
            else:
                inventory -= qty
                cash_cents += qty * px
                print(f"✅ FILL: sold {qty} YES @ {px}c")

    def kill_all_positions(tag="SESSION END"):
        """Cancel everything, then flatten with escalating aggression until
        verifiably flat. This is the 'nothing survives the demo' switch."""
        nonlocal cash_cents, fees_cents, inventory
        print(f"\n🛑 {tag}: canceling quotes and killing all positions...")
        client.cancel_all_resting(TICKER)
        if DRY_RUN:
            return True
        for attempt in range(12):
            process_new_fills()
            inventory = client.get_position(TICKER)
            if inventory == 0:
                print("✅ POSITION VERIFIED FLAT.")
                return True
            book = client.get_orderbook(TICKER)
            bid, ask, *_ = best_prices(book)
            qty = min(abs(inventory), 100)
            if inventory > 0:
                px = max(1, bid - attempt)          # walk down through the book
                client.place_limit(TICKER, "sell", qty, px)
            else:
                px = min(99, ask + attempt)         # walk up through the book
                client.place_limit(TICKER, "buy", qty, px)
            fees_cents += taker_fee_cents(qty, px)
            time.sleep(1.5)
            client.cancel_all_resting(TICKER)       # clear any unfilled remainder
        process_new_fills()
        inventory = client.get_position(TICKER)
        if inventory != 0:
            print(f"🚨 CRITICAL: still holding {inventory} after escalation. "
                  f"Close manually in the UI NOW.")
            return False
        print("✅ POSITION VERIFIED FLAT.")
        return True

    # make sure the kill switch runs even on Ctrl-C or a crash
    atexit.register(lambda: kill_all_positions("EXIT SAFETY"))

    # --- optional guaranteed opening fill (crosses the spread once) ---
    if STARTER_CROSS and not DRY_RUN:
        book = client.get_orderbook(TICKER)
        bid, ask, bid_sz, ask_sz = best_prices(book)
        fair = microprice(bid, ask, bid_sz, ask_sz)
        # open on the cheaper-looking side per microprice
        if fair >= (bid + ask) / 2:
            print(f"🎬 Starter: buying {STARTER_SIZE} YES at the ask ({ask}c)")
            client.place_limit(TICKER, "buy", STARTER_SIZE, ask)
            fees_cents += taker_fee_cents(STARTER_SIZE, ask)
        else:
            print(f"🎬 Starter: selling {STARTER_SIZE} YES at the bid ({bid}c)")
            client.place_limit(TICKER, "sell", STARTER_SIZE, bid)
            fees_cents += taker_fee_cents(STARTER_SIZE, bid)
        time.sleep(2)
        process_new_fills()

    deadline = time.time() + RUN_MINUTES * 60
    next_report = time.time() + REPORT_EVERY_S
    cycle = 1
    print(f"\nQuoting {TICKER} for {RUN_MINUTES} minutes...")

    while time.time() < deadline:
        time.sleep(1)
        try:
            book = client.get_orderbook(TICKER)
        except requests.HTTPError as e:
            print(f"Waiting for market data... ({e})")
            continue
        bid, ask, bid_sz, ask_sz = best_prices(book)

        process_new_fills()

        fair = microprice(bid, ask, bid_sz, ask_sz)
        state = [bid / 100, ask / 100, bid_sz / 10000, ask_sz / 10000,
                 fair / 100, inventory / MAX_INVENTORY]
        posture = agent.choose_action(state, inventory)
        qbid, qask = make_quotes(bid, ask, bid_sz, ask_sz, inventory, posture)

        mark = cash_cents + inventory * fair - fees_cents
        color = "\033[92m" if mark >= 0 else "\033[91m"
        print(f"| {bid}c x {ask}c (micro {fair:5.2f}) | Inv {inventory:4d} | "
              f"Quote {qbid}/{qask} | PNL {color}${mark / 100:7.2f}\033[0m | {posture}")

        # refresh quotes only when the target moved (less churn, fewer requests)
        if inventory < MAX_INVENTORY:
            if resting["buy"][1] != qbid:
                client.cancel(resting["buy"][0])
                oid = client.place_limit(TICKER, "buy", ORDER_SIZE, qbid)
                resting["buy"] = (oid, qbid)
        else:
            client.cancel(resting["buy"][0]); resting["buy"] = (None, None)
        if inventory > -MAX_INVENTORY:
            if resting["sell"][1] != qask:
                client.cancel(resting["sell"][0])
                oid = client.place_limit(TICKER, "sell", ORDER_SIZE, qask)
                resting["sell"] = (oid, qask)
        else:
            client.cancel(resting["sell"][0]); resting["sell"] = (None, None)

        if time.time() >= next_report:
            print(f"\n---> Cycle {cycle} | fills so far: {n_fills} | "
                  f"est. fees: ${fees_cents / 100:.2f} | "
                  f"PNL (marked to micro): ${mark / 100:.2f}\n")
            cycle += 1
            next_report += REPORT_EVERY_S

    # --- end of session: kill everything, then report realized numbers ---
    ok = kill_all_positions()
    process_new_fills()
    realized = cash_cents - fees_cents   # flat => cash IS the realized pnl
    print("\n" + "-" * 70)
    print(f"Demo finished. Fills: {n_fills} | Est. fees: ${fees_cents / 100:.2f}")
    print(f"\033[1mFinal realized PNL: ${realized / 100:.2f}\033[0m"
          + ("" if ok else "  (WARNING: position not flat!)"))


# --- 5. Market scanner -------------------------------------------------------

def scan(client, query):
    """Rank live markets matching `query` by fillability: volume and spread.
    Run this the morning of the talk and pick the liveliest book."""
    print(f"Scanning open markets matching {query!r}...")
    markets = client.list_open_markets(query)[:15]
    print(f"{'TICKER':<28}{'VOL':>12}{'BOOK':>12}{'SPRD':>6}  TITLE")
    for m in markets:
        try:
            book = client.get_orderbook(m["ticker"])
            bid, ask, bid_sz, ask_sz = best_prices(book)
            spread = ask - bid
        except requests.HTTPError:
            bid = ask = spread = 0
        print(f"{m['ticker']:<28}{(m.get('volume') or 0):>12,}"
              f"{f'{bid}x{ask}':>12}{spread:>5}c  {m.get('title','')[:48]}")
        time.sleep(0.2)
    print("\nPick: enough volume to print in 20 minutes, widest spread you can "
          "get. 2c+ on an unflagged (no-maker-fee) series is the sweet spot.")


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    if not KEY_ID:
        print("Set KALSHI_ACCESS_KEY and KALSHI_PRIVATE_KEY_PATH first.")
        sys.exit(1)

    client = KalshiClient(BASE_URL, KEY_ID, PRIVATE_KEY_PATH)

    if len(sys.argv) > 2 and sys.argv[1] == "--scan":
        scan(client, sys.argv[2])
        sys.exit(0)

    m = client.get_market(TICKER)
    print(f"Market: {m['title']}  [{TICKER}]")
    print(f"Last: {m.get('last_price')}c | Volume: {m.get('volume'):,} contracts")
    if SHOW_BALANCE:
        print(f"Balance: ${client.get_balance() / 100:,.2f}")

    agent = Agent(state_size=6, epsilon=0.2)
    run_session(client, agent)
