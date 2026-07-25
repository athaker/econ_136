"""
Kalshi Market-Making Demo — Agents for Finance (2026)
Avi Thaker & Daniel Sonner

A simple Q-learning agent that MAKES markets on a Kalshi event contract:
it posts passive YES bids/asks one tick inside the touch, collects fills,
and tries to earn the spread while managing inventory.

DEMO ONLY. NOT FINANCIAL ADVICE. Please don't try this at home.

Setup:
  1. Kalshi -> Settings -> API -> create an API key (you get a key ID and
     download an RSA private key .pem).
  2. export KALSHI_ACCESS_KEY="your-key-id"
     export KALSHI_PRIVATE_KEY_PATH="/path/to/kalshi_private_key.pem"
  3. pip install requests cryptography torch

Prices are in CENTS (1-99). A contract settles at 100c (YES) or 0c (NO),
so max loss per contract is known up front.
"""

import atexit
import base64
import os
import random
import time
import uuid

import requests
import torch
import torch.nn as nn
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# --- Configuration -----------------------------------------------------------

# 2026 House control (Democratic side) — the most liquid midterm "results"
# market. Any open ticker works; run with --list <query> to search markets.
TICKER = "CONTROLH-2026-D"

# Production API. For a rehearsal without real money, use the demo env:
#   https://demo-api.kalshi.co/trade-api/v2  (separate demo account/keys)
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

ORDER_SIZE = 10        # contracts per quote (max loss <= $0.99 each)
MAX_INVENTORY = 30     # absolute net YES position cap
CYCLES = 25            # agent cycles for the session
SECONDS_PER_CYCLE = 45
DRY_RUN = False        # True = print orders instead of sending them

KEY_ID = os.environ.get("KALSHI_ACCESS_KEY", "")
PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")


# --- 1. Q-Learning Agent and Neural Network (same brain, new venue) ----------

class QNetwork(nn.Module):
    """A simple neural network to map market state to action Q-values."""

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
    """Epsilon-greedy agent. Actions now place PASSIVE quotes, not takes."""

    def __init__(self, state_size, action_size, epsilon=0.2):
        self.q_network = QNetwork(state_size, action_size)
        self.actions_map = {0: "HOLD", 1: "BID", 2: "ASK"}
        self.epsilon = epsilon
        print(f"Agent initialized with epsilon = {epsilon}. "
              f"It will explore {epsilon * 100:.0f}% of the time.")

    def choose_action(self, state, inventory):
        valid = ["HOLD"]
        if inventory < MAX_INVENTORY:
            valid.append("BID")    # post a resting buy (YES) inside the spread
        if inventory > -MAX_INVENTORY:
            valid.append("ASK")    # post a resting sell (YES) inside the spread

        if random.random() < self.epsilon:
            action = random.choice(valid)
            print(f"🎲 Exploring: Chose {action}")
            return action
        with torch.no_grad():
            q = self.q_network(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
            action = self.actions_map[torch.argmax(q).item()]
            if action not in valid:
                action = "HOLD"
            print(f"🧠 Exploiting: Chose {action}")
            return action


# --- 2. Minimal Kalshi REST client (RSA-PSS signed requests) -----------------

class KalshiClient:
    def __init__(self, base_url, key_id, private_key_path):
        self.base_url = base_url
        self.key_id = key_id
        with open(private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)
        self.session = requests.Session()

    def _headers(self, method, path):
        # Sign: timestamp_ms + METHOD + path (incl. /trade-api/v2, no query string)
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
        r = self.session.request(
            method, self.base_url + path, params=params, json=json,
            headers=self._headers(method, api_path), timeout=10,
        )
        r.raise_for_status()
        return r.json() if r.text else {}

    # -- market data --
    def get_market(self, ticker):
        return self._request("GET", f"/markets/{ticker}")["market"]

    def get_orderbook(self, ticker, depth=5):
        return self._request("GET", f"/markets/{ticker}/orderbook",
                             params={"depth": depth})["orderbook"]

    def search_markets(self, query):
        out, cursor = [], None
        for _ in range(10):
            params = {"status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            resp = self._request("GET", "/markets", params=params)
            for m in resp.get("markets", []):
                text = (m.get("title", "") + " " + m["ticker"]).lower()
                if query.lower() in text:
                    out.append((m["ticker"], m.get("title", ""), m.get("volume", 0)))
            cursor = resp.get("cursor")
            if not cursor:
                break
        return sorted(out, key=lambda t: -t[2])

    # -- trading --
    def place_limit(self, ticker, action, count, yes_price):
        if DRY_RUN:
            print(f"[DRY RUN] {action} {count} YES @ {yes_price}c")
            return None
        order = self._request("POST", "/portfolio/orders", json={
            "client_order_id": str(uuid.uuid4()),
            "ticker": ticker,
            "action": action,          # "buy" or "sell"
            "side": "yes",             # we quote in YES-space only
            "type": "limit",
            "count": count,
            "yes_price": yes_price,    # cents
        })
        return order["order"]["order_id"]

    def cancel(self, order_id):
        try:
            self._request("DELETE", f"/portfolio/orders/{order_id}")
        except requests.HTTPError:
            pass  # already filled/canceled

    def cancel_all_resting(self, ticker):
        resp = self._request("GET", "/portfolio/orders",
                             params={"ticker": ticker, "status": "resting"})
        for o in resp.get("orders", []):
            self.cancel(o["order_id"])

    def get_fills(self, ticker, min_ts):
        resp = self._request("GET", "/portfolio/fills",
                             params={"ticker": ticker, "min_ts": min_ts, "limit": 200})
        return resp.get("fills", [])

    def get_position(self, ticker):
        resp = self._request("GET", "/portfolio/positions", params={"ticker": ticker})
        for p in resp.get("market_positions", []):
            if p["ticker"] == ticker:
                return p.get("position", 0)   # net YES contracts (negative = NO)
        return 0

    def get_balance(self):
        return self._request("GET", "/portfolio/balance")["balance"]


def best_prices(book):
    """Return (best_yes_bid, best_yes_ask, bid_size, ask_size) in cents.

    Kalshi books list resting YES bids and resting NO bids;
    a NO bid at price q is a YES ask at 100 - q.
    """
    yes = book.get("yes") or []
    no = book.get("no") or []
    best_bid, bid_sz = (yes[-1][0], yes[-1][1]) if yes else (1, 0)
    best_ask, ask_sz = (100 - no[-1][0], no[-1][1]) if no else (99, 0)
    return best_bid, best_ask, bid_sz, ask_sz


def taker_fee_cents(count, price_cents):
    """Kalshi general fee schedule: ceil(0.07 * C * P * (1-P)), maker fee 0
    on most series. Only our forced flattening pays this."""
    p = price_cents / 100.0
    import math
    return math.ceil(0.07 * count * p * (1 - p) * 100)


# --- 3. Main quoting loop ----------------------------------------------------

total_pnl_cents = 0.0

def get_state(bid, ask, bid_sz, ask_sz, inventory):
    return [bid / 100, ask / 100, bid_sz / 10000, ask_sz / 10000,
            inventory / MAX_INVENTORY]


def execute_quote_cycle(client, agent):
    """One cycle: the agent quotes for SECONDS_PER_CYCLE, then flattens."""
    global total_pnl_cents
    print("\n" + "-" * 70)
    print("Starting new agent cycle...")

    inventory = client.get_position(TICKER)
    cash_cents = 0.0                       # realized cash flow this cycle
    resting = {}                           # side -> (order_id, price)
    since_ts = int(time.time())
    seen_fills = set()

    def process_new_fills():
        nonlocal inventory, cash_cents
        for f in client.get_fills(TICKER, since_ts):
            fid = f.get("trade_id") or f.get("fill_id") or f.get("order_id", "") + str(f.get("created_time"))
            if fid in seen_fills:
                continue
            seen_fills.add(fid)
            qty, px = f["count"], f["yes_price"]
            if f["action"] == "buy":
                inventory += qty
                cash_cents -= qty * px
                print(f"✅ Filled: BOUGHT {qty} YES @ {px}c")
            else:
                inventory -= qty
                cash_cents += qty * px
                print(f"✅ Filled: SOLD {qty} YES @ {px}c")

    for _ in range(SECONDS_PER_CYCLE):
        time.sleep(1)
        try:
            book = client.get_orderbook(TICKER)
        except requests.HTTPError as e:
            print(f"Waiting for valid market data... ({e})")
            continue

        bid, ask, bid_sz, ask_sz = best_prices(book)
        if ask - bid < 1:
            continue

        # 1. Process any fills since last check
        process_new_fills()

        # 2. Ask the agent what to do
        state = get_state(bid, ask, bid_sz, ask_sz, inventory)
        action = agent.choose_action(state, inventory)

        mid = (bid + ask) / 2
        unrealized = cash_cents + inventory * mid
        color = "\033[92m" if unrealized >= 0 else "\033[91m"
        print(f"| Book {bid}c x {ask}c | Inv: {inventory:4d} | "
              f"Cycle PNL (mark-to-mid): {color}${unrealized / 100:8.2f}\033[0m | {action}")

        # 3. Quote passively, one tick inside the touch (never cross)
        if action == "BID":
            px = min(bid + 1, ask - 1)
            if resting.get("buy", (None, None))[1] != px:
                if "buy" in resting:
                    client.cancel(resting["buy"][0])
                oid = client.place_limit(TICKER, "buy", ORDER_SIZE, px)
                resting["buy"] = (oid, px)
                print(f"📌 Quoting BID {ORDER_SIZE} @ {px}c")
        elif action == "ASK":
            px = max(ask - 1, bid + 1)
            if resting.get("sell", (None, None))[1] != px:
                if "sell" in resting:
                    client.cancel(resting["sell"][0])
                oid = client.place_limit(TICKER, "sell", ORDER_SIZE, px)
                resting["sell"] = (oid, px)
                print(f"📌 Quoting ASK {ORDER_SIZE} @ {px}c")

    # --- End of cycle: pull quotes and flatten ---
    print("Cycle over. Canceling quotes and flattening inventory...")
    client.cancel_all_resting(TICKER)
    inventory = client.get_position(TICKER)

    for _ in range(5):
        if inventory == 0:
            break
        book = client.get_orderbook(TICKER)
        bid, ask, *_ = best_prices(book)
        qty = min(abs(inventory), ORDER_SIZE * 3)
        if inventory > 0:   # sell into the bid (marketable limit)
            client.place_limit(TICKER, "sell", qty, bid)
            fee = taker_fee_cents(qty, bid)
        else:               # buy from the ask
            client.place_limit(TICKER, "buy", qty, ask)
            fee = taker_fee_cents(qty, ask)
        cash_cents -= fee
        time.sleep(2)
        process_new_fills()
        inventory = client.get_position(TICKER)

    if inventory != 0:
        print(f"CRITICAL: could not fully flatten (inventory={inventory}). "
              f"Marking remainder to mid.")
        book = client.get_orderbook(TICKER)
        bid, ask, *_ = best_prices(book)
        cash_cents += inventory * (bid + ask) / 2

    print(f"✅ Cycle closed. \033[1mRealized PNL for this cycle: "
          f"${cash_cents / 100:.2f}\033[0m")
    total_pnl_cents += cash_cents


# --- Main Execution Block ----------------------------------------------------

if __name__ == "__main__":
    import sys

    if not KEY_ID:
        print("Set KALSHI_ACCESS_KEY and KALSHI_PRIVATE_KEY_PATH first.")
        sys.exit(1)

    client = KalshiClient(BASE_URL, KEY_ID, PRIVATE_KEY_PATH)

    if len(sys.argv) > 2 and sys.argv[1] == "--list":
        for tick, title, vol in client.search_markets(sys.argv[2])[:20]:
            print(f"{tick:<30} vol={vol:<12,} {title}")
        sys.exit(0)

    # Always leave the book clean, even on Ctrl-C
    atexit.register(lambda: client.cancel_all_resting(TICKER))

    m = client.get_market(TICKER)
    print(f"Market: {m['title']}  [{TICKER}]")
    print(f"Last: {m.get('last_price')}c | Volume: {m.get('volume'):,} contracts")
    print(f"Balance: ${client.get_balance() / 100:,.2f}")

    agent = Agent(state_size=5, action_size=3, epsilon=0.2)

    try:
        for i in range(CYCLES):
            execute_quote_cycle(client, agent)
            print(f"\n---> Running Total PNL after cycle {i + 1}/{CYCLES}: "
                  f"${total_pnl_cents / 100:.2f}")
            if i < CYCLES - 1:
                print("Pausing for 5 seconds before next cycle...")
                time.sleep(5)
    finally:
        client.cancel_all_resting(TICKER)

    print("\n" + "-" * 70)
    print(f"Trading demo finished. Final Total PNL: ${total_pnl_cents / 100:.2f}")
