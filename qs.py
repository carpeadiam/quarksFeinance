from quarks3 import *
from datetime import datetime, timedelta
import json


class QuarkScriptInterpreter:
    def __init__(self, user_id=None):
        self.user_id = user_id
        self.current_portfolio = None
        self.current_watchlist = None
        self.strategy_types = {
            'MOMENTUM': 'Momentum Strategy',
            'BOLLINGER': 'Bollinger Bands',
            'MACROSS': 'Moving Average Crossover'
        }

    def execute(self, command):
        parts = command.strip().split()
        if not parts:
            return False, "Empty command"

        cmd = parts[0].upper()
        args = " ".join(parts[1:])

        try:
            if cmd == "CREATE":
                return self._handle_create(args)
            elif cmd == "BUY":
                return self._handle_buy(args)
            elif cmd == "SELL":
                return self._handle_sell(args)
            elif cmd == "VIEW":
                return self._handle_view(args)
            elif cmd == "ADD":
                return self._handle_add(args)
            elif cmd == "STRATEGY":
                return self._handle_strategy(args)
            elif cmd == "RUN":
                return self._handle_run(args)
            elif cmd == "GENERATE":
                return self._handle_generate(args)
            elif cmd == "SAVE":
                return self._handle_save()
            elif cmd == "LOAD":
                return self._handle_load(args)
            elif cmd == "EXIT":
                print("Exiting QuarkScript.")
                return True, ""
            else:
                return False, f"Unknown command: {cmd}"

        except Exception as e:
            return False, f"Error executing command: {str(e)}"

    def _handle_create(self, args):
        if "PORTFOLIO" in args:
            params = self._parse_params(args.replace("PORTFOLIO", ""))
            if not self.user_id:
                return False, "You must be logged in to create a portfolio"
            self.current_portfolio = Simulation(params["name"], float(params["cash"]))
            return True, f"Portfolio '{params['name']}' created with ₹{params['cash']:.2f} cash"
        elif "WATCHLIST" in args:
            params = self._parse_params(args.replace("WATCHLIST", ""))
            if not self.user_id:
                return False, "You must be logged in to create a watchlist"
            self.current_watchlist = Watchlist(params["name"])
            return True, f"Watchlist '{params['name']}' created"
        else:
            return False, "Invalid CREATE command. Use CREATE PORTFOLIO or CREATE WATCHLIST"

    def _handle_buy(self, args):
        if not self.current_portfolio:
            return False, "No portfolio loaded. Use CREATE PORTFOLIO or LOAD PORTFOLIO first"

        params = self._parse_params(args)
        if "symbol" not in params or "quantity" not in params:
            return False, "BUY command requires symbol and quantity parameters"

        self.current_portfolio.buy_stock(params["symbol"], int(params["quantity"]))
        return True, f"Bought {params['quantity']} shares of {params['symbol']}"

    def _handle_strategy(self, args):
        subparts = args.split()
        if not subparts:
            return False, "Missing strategy subcommand"

        subcmd = subparts[0].upper()
        subargs = " ".join(subparts[1:])

        if subcmd == "CREATE":
            return self._strategy_create(subargs)
        elif subcmd == "LIST":
            return self._strategy_list()
        elif subcmd == "DELETE":
            return self._strategy_delete(subargs)
        elif subcmd == "ENABLE":
            return self._strategy_toggle(subargs, True)
        elif subcmd == "DISABLE":
            return self._strategy_toggle(subargs, False)
        else:
            return False, f"Unknown strategy command: {subcmd}"

    def _strategy_create(self, args):
        if not self.user_id:
            return False, "You must be logged in to create strategies"
        if not self.current_portfolio:
            return False, "No portfolio loaded. Load or create a portfolio first"

        params = self._parse_params(args)
        required = ["name", "symbol", "type"]
        if not all(k in params for k in required):
            return False, f"Missing required parameters: {', '.join(required)}"
        if params["type"].upper() not in self.strategy_types:
            return False, f"Invalid strategy type. Available: {', '.join(self.strategy_types.keys())}"

        # Create and save strategy
        strategy = {
            'user_id': self.user_id,
            'portfolio_id': self.current_portfolio.db_id,
            'name': params["name"],
            'symbol': params["symbol"].upper(),
            'strategy_type': params["type"].upper(),
            'parameters': {k: v for k, v in params.items() if k not in required}
        }

        if save_strategy(strategy):
            return True, (f"Created strategy '{params['name']}' for {params['symbol']}\n"
                          f"Type: {self.strategy_types[params['type'].upper()]}\n"
                          f"Portfolio: {self.current_portfolio.name}")
        return False, "Failed to save strategy"

    def _strategy_list(self):
        if not self.user_id:
            return False, "You must be logged in to list strategies"

        strategies = list_strategies(self.user_id)
        if not strategies:
            return True, "No strategies found"

        output = ["Your Strategies:"]
        for s in strategies:
            output.append(
                f"ID: {s['id']}, Name: {s['name']}\n"
                f"Symbol: {s['symbol']}, Type: {s['strategy_type']}\n"
                f"Portfolio: {s['portfolio_name']} (ID: {s['portfolio_id']})\n"
                f"Active: {'Yes' if s['is_active'] else 'No'}\n"
                f"Last run: {s['last_executed'] or 'Never'}\n"
                f"Params: {json.dumps(s['parameters'])}\n"
                "------"
            )
        return True, "\n".join(output)

    def _strategy_delete(self, args):
        if not self.user_id:
            return False, "You must be logged in to delete strategies"

        params = self._parse_params(args)
        if "id" not in params:
            return False, "Missing strategy ID"

        if delete_strategy(self.user_id, params["id"]):
            return True, f"Deleted strategy {params['id']}"
        return False, "Failed to delete strategy"

    def _strategy_toggle(self, args, active):
        if not self.user_id:
            return False, "You must be logged in to modify strategies"

        params = self._parse_params(args)
        if "id" not in params:
            return False, "Missing strategy ID"

        if toggle_strategy(self.user_id, params["id"], active):
            status = "enabled" if active else "disabled"
            return True, f"Strategy {params['id']} {status}"
        return False, "Failed to update strategy"

    # ... (keep all other existing _handle_* methods exactly as they were) ...

    def _parse_params(self, args):
        """Parse key-value pairs from command arguments."""
        params = {}
        for pair in args.split():
            if "=" in pair:
                key, value = pair.split("=", 1)
                params[key.strip()] = value.strip().strip('"\'')
        return params


def quark_script_main(user_id=None):
    interpreter = QuarkScriptInterpreter(user_id)
    print("Welcome to QuarkScript! Type your commands below.")
    print("Type 'HELP' for available commands.")

    while True:
        try:
            command = input("QuarkScript> ")
            if command.upper() == "HELP":
                print_help()
                continue

            success, message = interpreter.execute(command)
            print(message)
            if success and command.upper().startswith("EXIT"):
                break

        except KeyboardInterrupt:
            print("\nExiting QuarkScript.")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def print_help():
    help_text = """
Available Commands:

Portfolio Management:
  CREATE PORTFOLIO name=<name> cash=<amount>
  LOAD PORTFOLIO id=<portfolio_id>
  VIEW PORTFOLIO
  BUY symbol=<symbol> quantity=<shares>
  SELL symbol=<symbol> quantity=<shares>
  SAVE

Watchlist Management:
  CREATE WATCHLIST name=<name>
  LOAD WATCHLIST id=<watchlist_id>
  VIEW WATCHLIST
  ADD TO WATCHLIST symbol=<symbol> notes=<optional_notes>
  SAVE

Strategy Management:
  STRATEGY CREATE name=<name> symbol=<symbol> type=<MOMENTUM|BOLLINGER|MACROSS> [params...]
  STRATEGY LIST
  STRATEGY DELETE id=<strategy_id>
  STRATEGY ENABLE id=<strategy_id>
  STRATEGY DISABLE id=<strategy_id>
  RUN STRATEGY name=<strategy_name> symbol=<symbol> [params...]

Analysis:
  GENERATE ADVICE symbol=<symbol>

Other:
  EXIT
"""
    print(help_text)


# Run the interpreter
if __name__ == "__main__":
    # For testing without auth:
    quark_script_main()

    # In production, you would first authenticate:
    # user_id = authenticate(username, password)
    # if user_id:
    #     quark_script_main(user_id)
