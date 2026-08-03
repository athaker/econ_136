"""
Diagnose Kalshi API auth. Run this before anything else:

    python check_auth.py

It never places an order. It checks, in order: the env vars, the private key
file, the signature against a public endpoint, and then authenticated endpoints
on BOTH the production and demo hosts — because a key minted in the demo
environment returns exactly this 401 against production, and vice versa.
"""

import base64
import os
import sys
import time

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

HOSTS = {
    "production": "https://api.elections.kalshi.com/trade-api/v2",
    "demo":       "https://demo-api.kalshi.co/trade-api/v2",
}

OK, BAD, WARN = "\033[92m✓\033[0m", "\033[91m✗\033[0m", "\033[93m!\033[0m"


def main():
    import kalshi_client as kc
    key_id = kc.load_key_id()
    key_path = kc.load_key_path()
    source = ("$KALSHI_ACCESS_KEY" if (os.environ.get("KALSHI_ACCESS_KEY") or "").strip()
              not in ("", "your-key-id", "<uuid>", "paste-the-uuid-here")
              else "key_id.txt")
    problems = []

    # --- 1. credentials ---
    print("1. Credentials")
    if not key_id:
        print(f"   {BAD} No key ID found.")
        print(f"     Either write it to key_id.txt (one line), or export "
              f"KALSHI_ACCESS_KEY.")
        problems.append("no key ID")
    elif key_id in ("your-key-id", "<uuid>", "paste-the-uuid-here", "your-key"):
        print(f"   {BAD} KALSHI_ACCESS_KEY is still the placeholder {key_id!r}.")
        problems.append("KALSHI_ACCESS_KEY is a placeholder")
    else:
        masked = key_id[:8] + "..." + key_id[-4:] if len(key_id) > 14 else key_id
        print(f"   {OK} key ID = {masked}  (len {len(key_id)}, from {source})")
        looks_uuid = len(key_id) == 36 and key_id.count("-") == 4
        if not looks_uuid:
            print(f"   {WARN} That is not shaped like a UUID. Kalshi key IDs look "
                  f"like 1a2b3c4d-5e6f-7890-abcd-ef1234567890.")
            print(f"     Make sure this is the KEY ID from Settings -> API, not "
                  f"the private key, an email, or an account number.")
            problems.append("key ID is not UUID-shaped")
    print(f"   · KALSHI_PRIVATE_KEY_PATH = {key_path}")

    # --- 2. private key ---
    print("\n2. Private key file")
    if not os.path.exists(key_path):
        print(f"   {BAD} No such file: {key_path}")
        sys.exit(1)
    try:
        with open(key_path, "rb") as f:
            raw = f.read()
        pk = serialization.load_pem_private_key(raw, password=None)
    except Exception as e:
        print(f"   {BAD} Could not parse as a PEM private key: {e}")
        sys.exit(1)
    if not isinstance(pk, rsa.RSAPrivateKey):
        print(f"   {BAD} Not an RSA key ({type(pk).__name__}). Kalshi requires RSA.")
        sys.exit(1)
    print(f"   {OK} RSA private key, {pk.key_size}-bit")

    # --- 3. public endpoint (no auth) ---
    print("\n3. Public endpoint (proves network + host are fine)")
    try:
        r = requests.get(HOSTS["production"] + "/exchange/status", timeout=15)
        r.raise_for_status()
        print(f"   {OK} exchange/status -> {r.json().get('trading_active')=}")
    except Exception as e:
        print(f"   {BAD} {e}")

    # --- 4. authenticated, both hosts ---
    def signed_get(base, path):
        ts = str(int(time.time() * 1000))
        msg = f"{ts}GET/trade-api/v2{path}".encode()
        sig = pk.sign(msg,
                      padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                                  salt_length=padding.PSS.DIGEST_LENGTH),
                      hashes.SHA256())
        return requests.get(base + path, timeout=15, headers={
            "KALSHI-ACCESS-KEY": key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "Content-Type": "application/json",
        })

    print("\n4. Authenticated endpoint on each host")
    good_host = None
    codes = set()
    for name, base in HOSTS.items():
        try:
            r = signed_get(base, "/portfolio/balance")
        except Exception as e:
            print(f"   {BAD} {name:<11} network error: {e}")
            continue
        try:
            codes.add(r.json().get("error", {}).get("details")
                      or r.json().get("error", {}).get("code"))
        except Exception:
            pass
        if r.status_code == 200:
            body = r.json()
            bal = body.get("balance_dollars")
            if bal is None:
                bal = f"{body.get('balance', 0) / 100:.2f}"
            print(f"   {OK} {name:<11} AUTH OK — balance ${float(bal):,.2f}")
            good_host = (name, base)
        else:
            print(f"   {BAD} {name:<11} {r.status_code} {r.text[:120]}")

    # --- verdict ---
    print("\n" + "-" * 70)
    if good_host:
        name, base = good_host
        print(f"Auth works against the {name.upper()} host.")
        if name == "demo":
            print("\nYour key is a DEMO key. The demo environment is paper money —\n"
                  "your $2.50 is NOT there. To use it, set in the scripts:\n"
                  f'    BASE_URL = "{base}"\n'
                  "Or generate a production key at kalshi.com -> Settings -> API.")
        else:
            print("You are ready to run:\n"
                  "    python book_viewer.py --scan\n"
                  "    python kalshi-market-making-demo.py TICKER --dry-run")
        return

    print("Auth FAILED on both hosts.\n")

    # Kalshi distinguishes these two, and they mean very different things.
    if "INVALID_PARAMETER" in codes:
        print("Kalshi returned INVALID_PARAMETER, which means a key ID WAS sent\n"
              "and the server rejected its FORMAT — it is not a well-formed key\n"
              "ID at all. (A wrong-but-valid key, or a bad signature, returns\n"
              "token_authentication_failure instead.)\n")
        print("Almost always this is a leftover placeholder or a stray quote in\n"
              "the exported value. Check what your shell actually has:\n")
        print('    echo "[$KALSHI_ACCESS_KEY]"\n')
        print("If that prints [your-key-id], [<uuid>], or anything with quotes or\n"
              "spaces inside the brackets, re-export it with the real UUID.\n")
        return
    if "token_authentication_failure" in codes:
        print("Kalshi returned token_authentication_failure: the key ID is\n"
              "well-formed but the credential did not validate — the ID is\n"
              "unknown/revoked, or key.txt is not its matching private key.\n")

    print("The signature scheme in kalshi_client.py matches Kalshi's spec, so\n"
          "the usual causes are, in order:\n")
    print("  1. The key ID is not the right value. In Settings -> API, the key ID\n"
          "     is the UUID displayed next to the key. It is NOT the downloaded\n"
          "     file. If you did not copy it, you cannot recover it — delete the\n"
          "     key and generate a new one (you get a fresh private key too).")
    print("  2. key.txt is not the private key that PAIRS with that key ID.\n"
          "     If you have generated keys more than once, the file on disk may\n"
          "     belong to an older, deleted key.")
    print("  3. The key was revoked, or the account is not API-enabled.")
    print("\nFix: generate a fresh key pair, save the new private key over\n"
          "key.txt, copy the key ID immediately, then re-run this script.")
    if problems:
        print("\nAlso noted: " + "; ".join(problems))


if __name__ == "__main__":
    main()
