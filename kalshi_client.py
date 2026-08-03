"""
Minimal Kalshi client for the CURRENT (post-March-2026) API.

Kalshi removed the legacy integer-cent fields in the Q1 2026 migration:
  * prices  are now fixed-point DOLLAR strings   ("0.8300")
  * counts  are now fixed-point strings          ("1.00", and may be fractional)
  * the orderbook payload is `orderbook_fp` with `yes_dollars` / `no_dollars`
  * orders moved to a single bid/ask book at POST /portfolio/events/orders
  * `tick_size` is gone; `price_ranges[].step` is canonical, and ticks are
    TAPERED — 0.1c steps below 10c and above 90c, 1c steps in between

Internally this module works in MILS (integer thousandths of a dollar, i.e.
deci-cents, 1-999) so all price math is exact integer math. Convert at the edges
with mils() / to_price().
"""

import base64
import math
import os
import time
import uuid

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

API_PREFIX = "/trade-api/v2"


def load_key_id():
    """Key ID from $KALSHI_ACCESS_KEY, else key_id.txt.

    The key ID is NOT a secret — it rides in a plaintext header on every
    request — but it is also not recoverable from the private key, so it is
    worth writing down once. A file beats an export that a stale shell can
    silently keep overriding.
    """
    kid = (os.environ.get("KALSHI_ACCESS_KEY") or "").strip()
    if kid and kid not in ("your-key-id", "<uuid>", "paste-the-uuid-here"):
        return kid
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "key_id.txt"), "key_id.txt"):
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line
    return kid  # may be "" or the placeholder; callers report on it


def load_key_path():
    p = (os.environ.get("KALSHI_PRIVATE_KEY_PATH") or "").strip()
    if p:
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, "key.txt")
    return local if os.path.exists(local) else "key.txt"


# --- price units --------------------------------------------------------------

def mils(dollars):
    """'0.8300' or 0.83 -> 830 (integer thousandths of a dollar)."""
    return int(round(float(dollars) * 1000))


def to_price(m):
    """830 -> '0.8300' (the string format the API wants)."""
    return f"{m / 1000:.4f}"


def cents(m):
    """830 -> '83.0c' for display."""
    return f"{m / 10:g}c"


def to_count(n):
    """1 -> '1.00' (fixed-point contract count)."""
    return f"{float(n):.2f}"


def taker_fee_dollars(count, price_mils):
    """Kalshi taker fee: ceil(0.07 * C * P * (1-P)) rounded up to the cent."""
    p = price_mils / 1000.0
    return math.ceil(0.07 * count * p * (1 - p) * 100) / 100.0


# --- tick structure -----------------------------------------------------------

def tick_mils(market, price_mils):
    """Valid tick size (in mils) at a given price for this market.

    `price_ranges` looks like [{start:'0.0000', end:'0.1000', step:'0.0010'}, ...]
    """
    ranges = market.get("price_ranges") or []
    for r in ranges:
        lo, hi = mils(r["start"]), mils(r["end"])
        if lo <= price_mils <= hi:
            return max(1, mils(r["step"]))
    return 10  # sane default: 1 cent


def round_to_tick(market, price_mils, direction=0):
    """Snap a price to a legal tick. direction: -1 down, +1 up, 0 nearest."""
    step = tick_mils(market, price_mils)
    if direction < 0:
        snapped = (price_mils // step) * step
    elif direction > 0:
        snapped = -((-price_mils) // step) * step
    else:
        snapped = int(round(price_mils / step)) * step
    return max(1, min(999, snapped))


# --- order book ---------------------------------------------------------------

def normalize(book):
    """`orderbook_fp` -> ({bid_mils: size}, {ask_mils: size}), both YES prices.

    `yes_dollars` are resting YES bids. `no_dollars` are resting NO bids; a NO
    bid at Q is a YES ask at (1.00 - Q).
    """
    bids, asks = {}, {}
    for px, sz in (book.get("yes_dollars") or []):
        m, s = mils(px), float(sz)
        if s > 0:
            bids[m] = bids.get(m, 0.0) + s
    for px, sz in (book.get("no_dollars") or []):
        m, s = 1000 - mils(px), float(sz)
        if s > 0:
            asks[m] = asks.get(m, 0.0) + s
    return bids, asks


def touch(bids, asks):
    """(best_bid, best_ask, bid_size, ask_size) in mils / contracts."""
    best_bid = max(bids) if bids else 1
    best_ask = min(asks) if asks else 999
    return best_bid, best_ask, bids.get(best_bid, 0.0), asks.get(best_ask, 0.0)


def microprice(bid, ask, bid_sz, ask_sz):
    """Size-weighted mid, in mils. Leans toward the side with less size."""
    if bid_sz + ask_sz <= 0:
        return (bid + ask) / 2
    return (bid * ask_sz + ask * bid_sz) / (bid_sz + ask_sz)


# --- client -------------------------------------------------------------------

class KalshiClient:
    def __init__(self, base_url, key_id, private_key_path):
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        with open(private_key_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(f.read(), password=None)
        self.session = requests.Session()

    def _headers(self, method, api_path):
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method}{api_path}".encode()
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
        # the signature covers the path WITHOUT the query string
        r = self.session.request(
            method, self.base_url + path, params=params, json=json,
            headers=self._headers(method, API_PREFIX + path), timeout=15)
        if r.status_code >= 400:
            raise requests.HTTPError(f"{r.status_code} {method} {path}: {r.text[:300]}",
                                     response=r)
        return r.json() if r.text else {}

    # ---- market data (public) ----
    def get_market(self, ticker):
        return self._request("GET", f"/markets/{ticker}")["market"]

    def get_orderbook(self, ticker, depth=10):
        resp = self._request("GET", f"/markets/{ticker}/orderbook",
                             params={"depth": depth})
        return resp.get("orderbook_fp") or resp.get("orderbook") or {}

    def recent_trades(self, ticker=None, limit=100):
        p = {"limit": limit}
        if ticker:
            p["ticker"] = ticker
        return self._request("GET", "/markets/trades", params=p).get("trades", [])

    # ---- portfolio ----
    def get_balance_dollars(self):
        b = self._request("GET", "/portfolio/balance")
        if b.get("balance_dollars") is not None:
            return float(b["balance_dollars"])
        return float(b.get("balance", 0)) / 100.0

    def get_position(self, ticker):
        """Net YES contracts held (float; fractional contracts are possible)."""
        resp = self._request("GET", "/portfolio/positions", params={"ticker": ticker})
        for p in resp.get("market_positions", []):
            if p.get("ticker") == ticker or p.get("market_ticker") == ticker:
                v = p.get("position_fp", p.get("position", 0))
                return float(v)
        return 0.0

    def get_fills(self, ticker, min_ts, limit=200):
        return self._request("GET", "/portfolio/fills",
                             params={"ticker": ticker, "min_ts": min_ts,
                                     "limit": limit}).get("fills", [])

    def get_resting_orders(self, ticker):
        """V2 /orders, falling back to the legacy portfolio path."""
        for path in ("/orders", "/portfolio/orders"):
            try:
                resp = self._request("GET", path,
                                     params={"ticker": ticker, "status": "resting"})
                return resp.get("orders", [])
            except requests.RequestException:
                continue
        return []

    # ---- trading (V2 single bid/ask book) ----
    def place_order(self, ticker, side, count, price_mils, post_only=True):
        """side: 'bid' (buy YES) or 'ask' (sell YES). price is the YES price."""
        body = {
            "ticker": ticker,
            "side": side,
            "count": to_count(count),
            "price": to_price(price_mils),
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "maker",
            "client_order_id": str(uuid.uuid4()),
        }
        if post_only:
            body["post_only"] = True          # guarantees maker (no taker fee)
        resp = self._request("POST", "/portfolio/events/orders", json=body)
        return resp.get("order_id")

    def cancel(self, order_id):
        if not order_id:
            return
        try:
            self._request("DELETE", f"/portfolio/events/orders/{order_id}")
        except requests.RequestException:
            try:
                self._request("DELETE", f"/portfolio/orders/{order_id}")
            except requests.RequestException:
                pass                          # already filled or gone

    def cancel_all(self, ticker):
        for o in self.get_resting_orders(ticker):
            self.cancel(o.get("order_id"))
