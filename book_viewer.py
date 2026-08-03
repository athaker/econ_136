"""
Text-mode order book viewer for Kalshi markets.

Renders a depth ladder — crowd bids below, crowd asks above, price down the
middle — plus a marker for where WE are in the book: our resting quotes, our
inventory, and the fair value we're quoting around.

Standalone (VIEW ONLY — places no orders):

    export KALSHI_ACCESS_KEY="your-key-id"
    export KALSHI_PRIVATE_KEY_PATH="./key.txt"
    python book_viewer.py TICKER
    python book_viewer.py TICKER --depth 12
    python book_viewer.py --scan            # find liquid markets to watch

Market data is public, so --scan and the ladder work even before your key is
sorted; only the inventory line needs credentials.

Prices are handled in MILS (thousandths of a dollar = deci-cents) — see
kalshi_client.py for why.
"""

import os
import shutil
import sys
import time

import kalshi_client as kc

# --- ANSI ---------------------------------------------------------------------

RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN = "\033[96m"; MAGENTA = "\033[95m"; WHITE = "\033[97m"; GREY = "\033[90m"

BAR = "█"; HALF = "▌"; BAR_WIDTH = 24

MIN_TOUCH_SIZE = 2      # --scan: contracts required at BOTH touches for a
                        # market to count as having a real two-sided book

# Kalshi's own event categories. Left to right is roughly the order this demo
# cares about: the talk is about crowds forecasting events, and an election or
# a "will X say Y on air" market makes that point far better than a Dota 2 map.
PREFERRED_CATEGORIES = ["Elections", "Politics", "Mentions", "Entertainment",
                        "Social", "World", "Companies", "Science and Technology"]
ALLOWED_CATEGORIES = ["Economics", "Financials", "Health", "Commodities",
                      "Climate and Weather", "Transportation"]
BLOCKED_CATEGORIES = ["Sports", "Crypto"]   # --sports re-includes both

CATEGORY_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".category_cache.json")
CACHE_MAX_AGE_S = 24 * 3600


def _plain_len(s):
    out, i = 0, 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
        else:
            out += 1
            i += 1
    return out


def _bar(size, max_size, width=BAR_WIDTH):
    if max_size <= 0 or size <= 0:
        return ""
    units = (size / max_size) * width
    full = int(units)
    return BAR * full + (HALF if units - full >= 0.5 else "")


def _sz(n):
    """Contract counts may be fractional now."""
    return f"{n:,.0f}" if n >= 10 or n == int(n) else f"{n:,.2f}"


def _px(m):
    """Mils -> compact cents label ('83c', '9.5c')."""
    return f"{m / 10:g}"


# --- ladder -------------------------------------------------------------------

def render(book, ticker="", title="", depth=8, fair=None, step=10,
           my_bid=None, my_ask=None, my_bid_size=0, my_ask_size=0,
           inventory=0.0, pnl=None, posture=None, note="", events=None,
           balance=None, banner=None, pnl_liq=None):
    """Return the ladder as printable lines. Prices are in MILS.

    `banner` is a loud one-line reason we are not quoting (market closed,
    empty book, no balance). It is drawn where the ladder would be, so the
    display never silently freezes on a dead market.
    """
    bids, asks = kc.normalize(book)
    best_bid, best_ask, bid_sz, ask_sz = kc.touch(bids, asks)
    empty = not bids or not asks

    step = max(1, step)
    lo = min([best_bid - depth * step] + [p for p in (my_bid, my_ask) if p])
    hi = max([best_ask + depth * step] + [p for p in (my_bid, my_ask) if p])
    lo, hi = max(1, lo), min(999, hi)

    # price rows on the market's own tick grid
    rows = list(range(lo - lo % step, hi + step, step))
    rows = [p for p in rows if 1 <= p <= 999]
    # make sure real levels and our own quotes always get a row
    for p in list(bids) + list(asks) + [my_bid, my_ask]:
        if p and lo <= p <= hi and p not in rows:
            rows.append(p)
    rows = sorted(set(rows), reverse=True)

    max_size = max([bids.get(p, 0) for p in rows] + [asks.get(p, 0) for p in rows] + [0])
    width = min(shutil.get_terminal_size((120, 40)).columns, 128)
    L = []

    # header
    head = f"{BOLD}{WHITE} {ticker}{RESET}"
    if title:
        head += f" {DIM}{title[:62]}{RESET}"
    L.append(head)

    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2
    tot = bid_sz + ask_sz
    imb = (bid_sz - ask_sz) / tot if tot else 0.0
    icol = GREEN if imb > 0.15 else RED if imb < -0.15 else GREY

    if empty:
        # touch() would invent a 0.1c/99.9c book here; never draw that as if real
        L.append(f"{GREY}{'─' * (width - 1)}{RESET}")
        side = ("no bids and no asks" if not bids and not asks
                else "no asks" if not asks else "no bids")
        L.append(f"  {BOLD}{YELLOW}EMPTY BOOK{RESET} — {side}. "
                 f"Nothing to quote against.")
        if banner:
            L.append(f"  {BOLD}{RED}{banner}{RESET}")
        L.append(f"{GREY}{'─' * (width - 1)}{RESET}")
        L.append(f" {DIM}INVENTORY{RESET} {inventory:+g}"
                 + (f"  {DIM}bal{RESET} ${balance:,.2f}" if balance is not None else ""))
        if events:
            for e in events:
                L.append(f" {e}")
        if note:
            L.append(f" {DIM}{note}{RESET}")
        return L

    bar = (f" {DIM}bid{RESET} {GREEN}{_px(best_bid)}c{RESET}{DIM}x{_sz(bid_sz)}{RESET}"
           f"  {DIM}ask{RESET} {RED}{_px(best_ask)}c{RESET}{DIM}x{_sz(ask_sz)}{RESET}"
           f"  {DIM}spread{RESET} {YELLOW}{_px(spread)}c{RESET}"
           f"  {DIM}mid{RESET} {_px(mid)}c")
    if fair is not None:
        bar += f"  {DIM}fair{RESET} {CYAN}{fair / 10:.2f}c{RESET}"
    bar += f"  {DIM}imb{RESET} {icol}{imb:+.0%}{RESET}"
    if balance is not None:
        bar += f"  {DIM}bal{RESET} {WHITE}${balance:,.2f}{RESET}"
    L.append(bar)
    L.append(f"{GREY}{'─' * (width - 1)}{RESET}")
    L.append(f"{DIM}{'CROWD BIDS':>{BAR_WIDTH + 13}}  {'PRICE':^7}  "
             f"{'CROWD ASKS':<{BAR_WIDTH + 13}}{RESET}")

    placed_fair = False
    for price in rows:
        bsz, asz = bids.get(price, 0.0), asks.get(price, 0.0)

        # spread separator + fair-value marker, drawn once between ask and bid
        if not placed_fair and price <= best_bid and fair is not None:
            pos = (fair - best_bid) / spread if spread > 0 else 0.5
            slot = int(round(max(0.0, min(1.0, pos)) * 20))
            track = "╌" * slot + f"{CYAN}◆{RESET}{GREY}" + "╌" * (20 - slot)
            L.append(f"{GREY}{'':>{BAR_WIDTH + 13}}  {'':^7}  {track}{RESET} "
                     f"{CYAN}fair {fair / 10:.2f}c{RESET}"
                     f"{GREY}  ({_px(spread)}c wide){RESET}")
            placed_fair = True

        mine_l = f"{MAGENTA}◀ YOU {_sz(my_bid_size)}{RESET}" if my_bid == price else ""
        mine_r = f"{MAGENTA}YOU {_sz(my_ask_size)} ▶{RESET}" if my_ask == price else ""

        left = f"{mine_l} {_sz(bsz)} {GREEN}{_bar(bsz, max_size)}{RESET}" if bsz \
            else (f"{mine_l} " if mine_l else "")
        left = " " * max(0, BAR_WIDTH + 13 - _plain_len(left)) + left

        right = f"{RED}{_bar(asz, max_size)}{RESET} {_sz(asz)} {mine_r}" if asz \
            else (f" {mine_r}" if mine_r else "")

        if price == best_bid:
            cell = f"{BOLD}{GREEN}{_px(price):>5}c{RESET}"
        elif price == best_ask:
            cell = f"{BOLD}{RED}{_px(price):>5}c{RESET}"
        elif price in (my_bid, my_ask):
            cell = f"{MAGENTA}{_px(price):>5}c{RESET}"
        else:
            cell = f"{GREY}{_px(price):>5}c{RESET}"

        L.append(f"{left}  {GREY}│{RESET}{cell}{GREY}│{RESET}  {right}")

    L.append(f"{GREY}{'─' * (width - 1)}{RESET}")

    icol2 = GREEN if inventory > 0 else RED if inventory < 0 else GREY
    quote = "flat"
    if my_bid or my_ask:
        quote = (f"{GREEN}{_px(my_bid)}c{RESET}" if my_bid else f"{GREY}--{RESET}") + \
                f"{GREY}/{RESET}" + \
                (f"{RED}{_px(my_ask)}c{RESET}" if my_ask else f"{GREY}--{RESET}")
    foot = (f" {DIM}INVENTORY{RESET} {icol2}{BOLD}{inventory:+g}{RESET}"
            f"  {DIM}QUOTE{RESET} {quote}")
    if my_bid and my_ask:
        foot += f" {DIM}({_px(my_ask - my_bid)}c wide){RESET}"
    if pnl is not None:
        pc = GREEN if pnl >= 0 else RED
        foot += f"  {DIM}P&L mark{RESET} {pc}{BOLD}${pnl:+.2f}{RESET}"
    if pnl_liq is not None:
        lc = GREEN if pnl_liq >= 0 else RED
        # what you'd ACTUALLY realize crossing the spread right now. The gap
        # between this and the mark IS the market maker's inventory risk.
        foot += f"  {DIM}liq{RESET} {lc}{BOLD}${pnl_liq:+.2f}{RESET}"
    if posture:
        foot += f"  {DIM}POSTURE{RESET} {YELLOW}{posture}{RESET}"
    L.append(foot)
    if banner:
        L.append(f" {BOLD}{RED}{banner}{RESET}")

    if events:
        L.append(f"{GREY}{'─' * (width - 1)}{RESET}")
        for e in events:
            L.append(f" {e}")
    if note:
        L.append(f" {DIM}{note}{RESET}")
    return L


def draw(*a, **k):
    sys.stdout.write("\033[H\033[J" + "\n".join(render(*a, **k)) + "\n")
    sys.stdout.flush()


# --- scanner ------------------------------------------------------------------

def series_of(market_ticker):
    """'KXFOXNEWSMENTION-26AUG02-CONS' -> 'KXFOXNEWSMENTION'."""
    return market_ticker.split("-", 1)[0]


def categories_for(client, series_tickers):
    """series_ticker -> category, resolved per-series and cached on disk.

    /events?category=X is ignored by the API, and the bulk /events listing does
    not cover every series (tennis, UFC and soccer all came back Unknown that
    way, so sports leaked straight through the filter). GET /series/{ticker} is
    authoritative, so look up exactly the series we care about and cache the
    answer — a series' category never changes.
    """
    import json
    import concurrent.futures as cf

    cache = {}
    if os.path.exists(CATEGORY_CACHE):
        try:
            with open(CATEGORY_CACHE) as f:
                cache = json.load(f).get("map", {})
        except Exception:
            cache = {}

    missing = sorted(set(series_tickers) - set(cache))
    if missing:
        print(f"Resolving {len(missing)} new series categories...")

        def one(st):
            try:
                return st, client._request(
                    "GET", f"/series/{st}")["series"].get("category", "Unknown")
            except Exception:
                return st, "Unknown"

        with cf.ThreadPoolExecutor(10) as ex:
            for st, cat in ex.map(one, missing):
                cache[st] = cat
        try:
            with open(CATEGORY_CACHE, "w") as f:
                json.dump({"ts": time.time(), "map": cache}, f)
        except Exception:
            pass
    return cache


# Wide, on-theme families that do NOT settle for weeks. The recent-trade feed
# under-ranks these: a market that prints 30 times a day rarely shows up in a
# two-minute window, but it is exactly what you want to quote on stage.
FOCUS_SERIES = ["KXTRUMPMENTION", "KXTRUMPSAY", "KXMAMDANIMENTION",
                "KXFOXNEWSMENTION"]


def scan(client, limit=25, include_sports=False, only=None, series=None):
    """Rank markets that are ACTUALLY trading by maker edge net of exit fees.

    Uses the public recent-trades feed: `/markets?status=open` is dominated by
    tens of thousands of zero-volume parlay markets and is useless for this.
    """
    import collections, concurrent.futures as cf

    print("Pulling recent exchange-wide trade flow...")
    trades, cursor = [], None
    for _ in range(6):
        p = {"limit": 1000}
        if cursor:
            p["cursor"] = cursor
        d = client._request("GET", "/markets/trades", params=p)
        trades += d.get("trades", [])
        cursor = d.get("cursor")
        if not cursor:
            break
    freq = collections.Counter(t["ticker"] for t in trades)

    if series:
        # Enumerate every open market in these series, regardless of whether it
        # happened to print in the last two minutes. Activity falls back to a
        # 24h-volume proxy so the FILL score stays comparable.
        top = []
        for s in series:
            try:
                ms = client._request("GET", "/markets",
                                     params={"series_ticker": s, "status": "open",
                                             "limit": 200}).get("markets", [])
            except requests.RequestException as e:
                print(f"  {s}: {e}")
                continue
            for m in ms:
                mt = m["ticker"]
                top.append(mt)
                if mt not in freq:
                    freq[mt] = float(m.get("volume_24h_fp") or 0) / 100.0
        print(f"{len(top)} open markets across {len(series)} series")
    else:
        top = [t for t, _ in freq.most_common(400) if not t.startswith("KXMVE")]
    cats = categories_for(client, [series_of(t) for t in top])

    def category(mt):
        return cats.get(series_of(mt), "Unknown")

    def keep(mt):
        c = category(mt)
        if only:
            return c.lower() in [o.lower() for o in only]
        if c in BLOCKED_CATEGORIES and not include_sports:
            return False
        return True

    dropped = collections.Counter(category(t) for t in top if not keep(t))
    print(f"{len(trades)} trades across {len(freq)} markets. "
          f"Excluded: {', '.join(f'{c} {n}' for c, n in dropped.most_common()) or 'none'}"
          f"\nChecking books...\n")

    def stat(t):
        try:
            bids, asks = kc.normalize(client.get_orderbook(t, depth=1))
            if not bids or not asks:
                return None            # empty book: touch() would fake 0.1c/99.9c
            b, a, bs, asz = kc.touch(bids, asks)
            return t, b, a, bs, asz
        except Exception:
            return None

    # KXMVE* are auto-generated parlay markets — tens of thousands of them, with
    # no genuine two-sided book. They dominate any unfiltered ranking.
    cands = [t for t in top if keep(t)]
    with cf.ThreadPoolExecutor(10) as ex:
        books = [r for r in ex.map(stat, cands) if r]

    rows = []
    for t, b, a, bs, asz in books:
        spread = a - b
        mid = (a + b) / 2
        if spread <= 0 or not (50 <= mid <= 950):
            continue
        if bs < MIN_TOUCH_SIZE or asz < MIN_TOUCH_SIZE:
            continue                   # need a real crowd on BOTH sides
        half_c = spread / 2 / 10.0
        net = half_c - kc.taker_fee_dollars(1, mid) * 100
        # Fillability. Three things have to be true at once to get filled on
        # stage: there is edge to capture, the market actually prints, and the
        # queue ahead of you is short enough to reach the front of. A 1c-wide
        # market with 145k contracts resting scores ~0 no matter how liquid.
        queue = min(bs, asz)          # the thinner side fills first
        activity = freq[t]
        fill = max(0.0, net) * (activity ** 0.5) / (1.0 + queue / 10.0)
        rows.append((net, t, b, a, bs, asz, activity, category(t), fill, queue))

    # Rank by fillability first — a preferred category is a tiebreak, not a
    # reason to show an unfillable penny-wide market at the top of the list.
    def rank(r):
        c = r[7]
        pref = 0 if c in PREFERRED_CATEGORIES else 1 if c in ALLOWED_CATEGORIES else 2
        return (-r[8], pref, -r[0])
    rows.sort(key=rank)

    print(f"{'TICKER':<40}{'CATEGORY':<15}{'BID':>7}{'ASK':>7}{'SPRD':>6}"
          f"{'BIDSZ':>8}{'ASKSZ':>8}{'TRD':>5}{'NET/1':>8}{'FILL':>7}")
    for net, t, b, a, bs, asz, n, c, fill, queue in rows[:limit]:
        col = GREEN if net > 0 else GREY
        ccol = CYAN if c in PREFERRED_CATEGORIES else GREY
        fcol = GREEN if fill >= 1.0 else YELLOW if fill >= 0.3 else GREY
        print(f"{t[:39]:<40}{ccol}{c[:14]:<15}{RESET}{_px(b) + 'c':>7}"
              f"{_px(a) + 'c':>7}{_px(a - b) + 'c':>6}"
              f"{_sz(bs):>8}{_sz(asz):>8}{n:>5.0f}{col}{net:>+7.1f}c{RESET}"
              f"{fcol}{fill:>7.1f}{RESET}")
    print(f"\n{DIM}NET/1 = half-spread captured as maker, minus the taker fee to "
          f"flatten 1 contract.\n"
          f"FILL  = edge x sqrt(trades) / queue ahead of you. Higher is a wider,\n"
          f"        livelier, shallower book — the three things you need at once.\n"
          f"        Green (>1.0) should fill inside a 15-minute session.\n"
          f"Sports/Crypto excluded; --sports includes them, "
          f"--only Elections,Politics narrows further.{RESET}")
    return [r[1] for r in rows[:limit]]


# --- standalone ---------------------------------------------------------------

def _main():
    argv = sys.argv[1:]
    depth = 8
    if "--depth" in argv:
        depth = int(argv[argv.index("--depth") + 1])
    positional = [a for a in argv if not a.startswith("--")
                  and a != str(depth)]

    key_id = kc.load_key_id()
    key_path = kc.load_key_path()
    client = kc.KalshiClient("https://api.elections.kalshi.com/trade-api/v2",
                             key_id or "public", key_path)

    if "--scan" in argv:
        only = series = None
        if "--only" in argv:
            only = [c.strip() for c in argv[argv.index("--only") + 1].split(",")]
        if "--focus" in argv:
            series = FOCUS_SERIES
        if "--series" in argv:
            series = [s.strip() for s in argv[argv.index("--series") + 1].split(",")]
        scan(client, include_sports="--sports" in argv, only=only, series=series)
        return

    if not positional:
        print("usage: python book_viewer.py TICKER [--depth N]\n"
              "       python book_viewer.py --scan")
        sys.exit(1)
    ticker = positional[0]

    market = client.get_market(ticker)
    title = market.get("title", "")
    step = kc.tick_mils(market, kc.mils(market.get("yes_bid_dollars") or "0.5"))

    print("\033[?25l", end="")
    try:
        while True:
            book = client.get_orderbook(ticker, depth=depth + 6)
            b, a, bs, asz = kc.touch(*kc.normalize(book))
            fair = kc.microprice(b, a, bs, asz)
            inv, bal = 0.0, None
            if key_id:
                try:
                    inv = client.get_position(ticker)
                    bal = client.get_balance_dollars()
                except Exception:
                    pass
            draw(book, ticker=ticker, title=title, depth=depth, fair=fair,
                 step=step, inventory=inv, balance=bal,
                 note="VIEW ONLY — no orders are placed.  Ctrl-C to exit.")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?25h", end="")


if __name__ == "__main__":
    _main()
