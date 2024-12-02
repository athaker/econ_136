from ib_insync import *
import time

util.startLoop()
# Connect to TWS or IB Gateway
ib = IB()
ib.connect("127.0.0.1", 4008, clientId=1)  # Adjust host/port/clientId as needed

# Define the USD.JPY future contract
contract = Forex("USDJPY")
ib.qualifyContracts(contract)

# Track PNL
pnl = 0.0


def close_existing_positions():
    # Check for any open positions
    positions = ib.positions()
    for position in positions:
        if position.contract.symbol == contract.symbol:
            action = "SELL" if position.position > 0 else "BUY"
            quantity = abs(position.position)
            print(f"Closing existing position: {action} {quantity} {contract.symbol}")
            close_order = MarketOrder(action=action, totalQuantity=quantity)
            close_trade = ib.placeOrder(contract, close_order)
            ib.sleep(2)  # Allow time for the order to fill
            if close_trade.fills:
                print("Position closed.")
            else:
                print("Failed to close position.")


def print_market_data():
    ticker = ib.reqMktData(contract, snapshot=False)  # Ensure continuous updates
    print("Realtime Bid/Ask and Sizes:")
    for _ in range(10):  # Limit to 10 updates for demo
        ib.sleep(1)  # Allow time for data to refresh
        print(
            f"Bid: {ticker.bid}, Ask: {ticker.ask}, Bid Size: {ticker.bidSize}, Ask Size: {ticker.askSize}"
        )
    ib.cancelMktData(contract)


def execute_cycle():
    global pnl

    # Step 1: Place a market order
    market_order = MarketOrder(action="BUY", totalQuantity=100)
    market_trade = ib.placeOrder(contract, market_order)
    print("Market buy order placed.")
    ib.sleep(2)  # Allow some time for the order to fill

    # Get the average fill price
    if market_trade.fills:
        buy_price = market_trade.fills[0].execution.avgPrice
        print(
            f"Market order status: {market_trade.orderStatus.status} - Filled at: {buy_price}"
        )
    else:
        print("Market order not filled.")
        return

    # Step 2: Place a limit order at the ask price
    ticker = ib.reqMktData(contract, snapshot=False)  # Ensure continuous updates
    ib.sleep(2)  # Allow time to fetch market data

    if ticker.ask:  # Ensure we have valid ask price
        limit_price = ticker.ask + 0.02  # Place limit order slightly above ask price
        limit_order = LimitOrder(action="SELL", totalQuantity=100, lmtPrice=limit_price)
        limit_trade = ib.placeOrder(contract, limit_order)
        print(f"Limit sell order placed at ask price: {limit_price}")

        # Monitor the order for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30:
            ib.sleep(1)
            print_market_data()  # Print market data continuously
            print(f"Order status: {limit_trade.orderStatus.status}")

            if limit_trade.orderStatus.status == "Filled":
                sell_price = limit_trade.fills[0].execution.avgPrice
                pnl += sell_price - buy_price
                print(f"Order filled. PNL for this cycle: {sell_price - buy_price:.2f}")
                break
        else:
            # If not filled in 30 seconds, close position with a market order
            print(
                "Limit order not filled within 30 seconds. Closing position with market order."
            )
            close_order = MarketOrder(action="SELL", totalQuantity=100)
            close_trade = ib.placeOrder(contract, close_order)
            ib.sleep(2)  # Allow time for the order to fill

            if close_trade.fills:
                sell_price = close_trade.fills[0].execution.avgPrice
                pnl += sell_price - buy_price
                print(
                    f"Market order executed. PNL for this cycle: {sell_price - buy_price:.2f}"
                )
            else:
                print("Market order to close position not filled.")

            # Cancel the limit order
            print("Cancelling the limit order.")
            ib.cancelOrder(limit_order)
    else:
        print("No valid ask price available for limit order.")


# Close any existing positions
close_existing_positions()

# Execute multiple cycles
total_cycles = 2
for cycle in range(total_cycles):
    print(f"\nStarting cycle {cycle + 1}...")
    execute_cycle()
    print(f"Running PNL after cycle {cycle + 1}: {pnl:.2f}")

# Disconnect from IB
print(f"Final PNL: {pnl:.2f}")
ib.disconnect()
