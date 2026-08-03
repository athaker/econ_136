"""
Panic button. Cancels every resting order and closes every open position,
across ALL tickers — not just the one a demo was quoting.

    python flatten.py            # show what would happen, change nothing
    python flatten.py --go       # actually do it

Use this when a session was killed uncleanly, when the network dropped
mid-demo, or any time you want to prove to yourself the account is flat.
The in-session kill switch only knows about the ticker it was quoting;
this knows about everything.
"""

import sys
import time

import requests

import kalshi_client as kc

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
GREEN, RED, YELLOW, DIM, RESET = ("\033[92m", "\033[91m", "\033[93m",
                                  "\033[2m", "\033[0m")


def main():
    go = "--go" in sys.argv
    client = kc.KalshiClient(BASE_URL, kc.load_key_id(), kc.load_key_path())

    print(f"Balance: ${client.get_balance_dollars():,.2f}")
    if not go:
        print(f"{YELLOW}DRY RUN — nothing will be changed. Re-run with --go "
              f"to execute.{RESET}\n")

    # --- 1. resting orders ---
    orders = client.get_resting_orders(None) or []
    if not orders:
        # get_resting_orders filters by ticker; fetch unfiltered too
        try:
            orders = client._request(
                "GET", "/portfolio/orders",
                params={"status": "resting", "limit": 200}).get("orders", [])
        except requests.HTTPError as e:
            print(f"could not list orders: {e}")
            orders = []

    print(f"Resting orders: {len(orders)}")
    for o in orders:
        px = o.get("price_dollars") or o.get("yes_price_dollars") or "?"
        print(f"  {o.get('ticker')}  {o.get('book_side') or o.get('action')}  "
              f"{o.get('remaining_count_fp')} @ {px}")
        if go:
            client.cancel(o.get("order_id"))
    if go and orders:
        print(f"  {GREEN}canceled{RESET}")

    # --- 2. open positions ---
    try:
        resp = client._request("GET", "/portfolio/positions",
                               params={"limit": 200})
    except requests.HTTPError as e:
        print(f"could not list positions: {e}")
        return
    positions = [(p.get("ticker") or p.get("market_ticker"),
                  float(p.get("position_fp", 0) or 0))
                 for p in resp.get("market_positions", [])]
    positions = [(t, q) for t, q in positions if abs(q) > 0.005]

    print(f"\nOpen positions: {len(positions)}")
    for ticker, qty in positions:
        print(f"  {ticker}  {qty:+g}")

    if not go:
        if positions or orders:
            print(f"\n{YELLOW}Re-run with --go to cancel and close.{RESET}")
        else:
            print(f"\n{GREEN}Account is already flat.{RESET}")
        return

    # --- 3. close each position, walking through the book ---
    for ticker, qty in positions:
        print(f"\nClosing {ticker} ({qty:+g})...")
        try:
            market = client.get_market(ticker)
        except requests.HTTPError as e:
            print(f"  {RED}cannot read market: {e}{RESET}")
            continue
        if market.get("status") != "active":
            print(f"  {YELLOW}market is {market.get('status')} — it will settle "
                  f"on its own; nothing to do.{RESET}")
            continue

        for attempt in range(10):
            live = client.get_position(ticker)
            if abs(live) < 0.005:
                print(f"  {GREEN}flat{RESET}")
                break
            try:
                b, a, _, _ = kc.touch(*kc.normalize(client.get_orderbook(ticker)))
                tick = kc.tick_mils(market, b)
                if live > 0:
                    px = kc.round_to_tick(market, b - attempt * tick, -1)
                    client.place_order(ticker, "ask", abs(live), px,
                                       post_only=False)
                else:
                    px = kc.round_to_tick(market, a + attempt * tick, +1)
                    client.place_order(ticker, "bid", abs(live), px,
                                       post_only=False)
                print(f"  attempt {attempt + 1}: {abs(live):g} @ {kc.cents(px)}")
            except requests.HTTPError as e:
                print(f"  attempt {attempt + 1} rejected: {str(e)[:100]}")
            time.sleep(1.5)
            try:
                client.cancel_all(ticker)
            except Exception:
                pass
        else:
            print(f"  {RED}COULD NOT CLOSE — do it manually in the Kalshi UI.{RESET}")

    print(f"\nFinal balance: ${client.get_balance_dollars():,.2f}")
    leftover = [t for t, _ in positions if abs(client.get_position(t)) > 0.005]
    if leftover:
        print(f"{RED}STILL OPEN: {leftover}{RESET}")
    else:
        print(f"{GREEN}All positions closed.{RESET}")


if __name__ == "__main__":
    main()
