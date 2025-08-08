import torch
import torch.nn as nn
import time
import random
from ib_insync import *

# --- 1. Q-Learning Agent and Neural Network Definition ---

class QNetwork(nn.Module):
    """A simple neural network to map market state to action Q-values."""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    """The Q-learning agent that makes decisions based on the QNetwork."""
    def __init__(self, state_size, action_size, epsilon=0.2):
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size)
        self.actions_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        self.epsilon = epsilon
        print(f"Agent initialized with epsilon = {self.epsilon}. It will explore {self.epsilon*100}% of the time.")

    def choose_action(self, state, current_position):
        """Implements an epsilon-greedy strategy."""
        if current_position == 0: # Flat
            valid_actions = ['BUY', 'SELL', 'HOLD']
        elif current_position == 1: # Long
            valid_actions = ['SELL', 'HOLD']
        else: # Short
            valid_actions = ['BUY', 'HOLD']

        # Epsilon-Greedy logic
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
            print(f"🎲 Exploring: Chose {action}")
            return action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor).squeeze(0)
                chosen_action = self.actions_map[torch.argmax(q_values).item()]
                if chosen_action not in valid_actions:
                    chosen_action = 'HOLD'
                print(f"🧠 Exploiting: Chose {chosen_action}")
                return chosen_action

# --- 2. Main Application Logic ---

util.startLoop()

# Connect to TWS or IB Gateway
ib = IB()
try:
    ib.connect("127.0.0.1", 4008, clientId=1) # Adjust host/port/clientId
except ConnectionRefusedError:
    print("Connection failed. Is TWS or IB Gateway running and configured for API connections?")
    exit()

# Define the contract to trade
#contract = Forex("USDJPY")
contract = Stock("SPY", "SMART", "USD")
ib.qualifyContracts(contract)

# MODIFIED: Renamed to clearly indicate this is the total PNL for the session.
total_pnl = 0.0
agent = Agent(state_size=4, action_size=3, epsilon=0.2)


def get_current_state(ticker):
    """Gathers and normalizes market data to form the state tensor."""
    bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
    ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
    bid_size = ticker.bidSize if ticker.bidSize else 0
    ask_size = ticker.askSize if ticker.askSize else 0
    return [bid / 100, ask / 100, bid_size / 10000, ask_size / 10000]

def close_all_positions():
    """Closes any open positions for the specified contract."""
    positions = ib.positions()
    for position in positions:
        if position.contract.symbol == contract.symbol and position.position != 0:
            action = "SELL" if position.position > 0 else "BUY"
            quantity = abs(position.position)
            print(f"Closing initial position: {action} {quantity} {contract.symbol}")
            order = MarketOrder(action=action, totalQuantity=quantity)
            trade = ib.placeOrder(contract, order)
            ib.sleep(2)
            if trade.orderStatus.status == 'Filled':
                print("Initial position closed.")
            else:
                print("Failed to close initial position.")

def execute_trade_cycle():
    """Executes one full trade cycle, tracking both cycle and total PNL."""
    global total_pnl
    print("\n----------------------------------------------------------------------")
    print("Starting new agent cycle...")

    position = 0
    pos_size = 0
    entry_price = 0

    ticker = ib.reqMktData(contract, '', False, False)
    ib.sleep(2)

    for _ in range(45):
        ib.sleep(1)

        if not ticker.bid or not ticker.ask:
            print("Waiting for valid market data...")
            continue

        state = get_current_state(ticker)
        action = agent.choose_action(state, position)

        unrealized_pnl = 0.0
        pos_string = "FLAT"
        if position == 1:
            current_price = ticker.bid
            unrealized_pnl = (current_price - entry_price) * pos_size
            pos_string = "LONG"
        elif position == -1:
            current_price = ticker.ask
            unrealized_pnl = (entry_price - current_price) * abs(pos_size)
            pos_string = "SHORT"

        pnl_color_code = "\033[92m" if unrealized_pnl >= 0 else "\033[91m"
        print(f"| Position: {pos_string:<5} | Size: {pos_size:4d} | Unrealized PNL: {pnl_color_code}${unrealized_pnl:8.2f}\033[0m | Action: {action}")

        if position == 0 and action in ['BUY', 'SELL']:
            order = MarketOrder(action=action, totalQuantity=100)
            trade = ib.placeOrder(contract, order)
            ib.sleep(2)
            if trade.orderStatus.status == 'Filled':
                position = 1 if action == 'BUY' else -1
                pos_size = 100 * position
                entry_price = trade.fills[0].execution.avgPrice
                print(f"✅ Position opened: {'LONG' if position == 1 else 'SHORT'} @ {entry_price}")
            else:
                print("⚠️ Order failed to fill.")

        elif (position == 1 and action == 'SELL') or (position == -1 and action == 'BUY'):
            order = MarketOrder(action=action, totalQuantity=100)
            trade = ib.placeOrder(contract, order)
            ib.sleep(2)
            if trade.orderStatus.status == 'Filled':
                exit_price = trade.fills[0].execution.avgPrice

                # --- PNL Calculation ---
                # 1. Calculate the PNL for this specific cycle.
                cycle_pnl = (exit_price - entry_price) * pos_size if position == 1 else (entry_price - exit_price) * abs(pos_size)
                print(f"✅ Position closed @ {exit_price}. \033[1mRealized PNL for this cycle: ${cycle_pnl:.2f}\033[0m")

                # 2. Add the cycle's result to the total PNL for the entire session.
                total_pnl += cycle_pnl

                ib.cancelMktData(ticker)
                return
            else:
                print("⚠️ Close order failed to fill.")

    if position != 0:
        print("Agent failed to close position in time. Forcing closure.")
        close_action = "SELL" if position == 1 else "BUY"
        order = MarketOrder(action=close_action, totalQuantity=100)
        trade = ib.placeOrder(contract, order)
        ib.sleep(2)
        if trade.orderStatus.status == 'Filled':
            exit_price = trade.fills[0].execution.avgPrice
            # Calculate PNL for this forced-closure cycle
            cycle_pnl = (exit_price - entry_price) * pos_size if position == 1 else (entry_price - exit_price) * abs(pos_size)
            print(f"Force closed position @ {exit_price}. \033[1mPNL for this cycle: ${cycle_pnl:.2f}\033[0m")
            # Add to total PNL
            total_pnl += cycle_pnl
        else:
            print("CRITICAL: Failed to force-close position.")

    ib.cancelMktData(ticker)

# --- Main Execution Block ---
close_all_positions()

total_cycles = 25
for i in range(total_cycles):
    execute_trade_cycle()
    # After each cycle, print the cumulative total PNL.
    print(f"\n---> Running Total PNL after cycle {i + 1}/{total_cycles}: ${total_pnl:.2f}")
    if i < total_cycles - 1:
      print("Pausing for 5 seconds before next cycle...")
      time.sleep(5)

# --- Final Summary ---
print("\n----------------------------------------------------------------------")
print(f"Trading demo finished. Final Total PNL: ${total_pnl:.2f}")
ib.disconnect()