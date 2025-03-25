from quarks import *


class QuarkScriptInterpreter:
    def __init__(self):
        self.current_portfolio = None
        self.current_watchlist = None

    def execute(self, command):
        parts = command.split()
        if not parts:
            return

        cmd = parts[0].upper()
        args = " ".join(parts[1:])

        try:
            if cmd == "CREATE":
                if "PORTFOLIO" in args:
                    params = self._parse_params(args.replace("PORTFOLIO", ""))
                    self.current_portfolio = Simulation(params["name"], float(params["cash"]))
                    print(f"Portfolio '{params['name']}' created with ₹{params['cash']:.2f} cash.")
                elif "WATCHLIST" in args:
                    params = self._parse_params(args.replace("WATCHLIST", ""))
                    self.current_watchlist = Watchlist(params["name"])
                    print(f"Watchlist '{params['name']}' created.")
                else:
                    print("Invalid CREATE command. Use CREATE PORTFOLIO or CREATE WATCHLIST.")

            elif cmd == "BUY":
                params = self._parse_params(args)
                if self.current_portfolio:
                    self.current_portfolio.buy_stock(params["symbol"], int(params["quantity"]))
                else:
                    print("No portfolio loaded. Use CREATE PORTFOLIO or LOAD PORTFOLIO first.")

            elif cmd == "SELL":
                params = self._parse_params(args)
                if self.current_portfolio:
                    self.current_portfolio.sell_stock(params["symbol"], int(params["quantity"]))
                else:
                    print("No portfolio loaded. Use CREATE PORTFOLIO or LOAD PORTFOLIO first.")

            elif cmd == "VIEW":
                if "PORTFOLIO" in args:
                    if self.current_portfolio:
                        self.current_portfolio.view_portfolio()
                    else:
                        print("No portfolio loaded.")
                elif "WATCHLIST" in args:
                    if self.current_watchlist:
                        self.current_watchlist.view_watchlist()
                    else:
                        print("No watchlist loaded.")
                else:
                    print("Invalid VIEW command. Use VIEW PORTFOLIO or VIEW WATCHLIST.")

            elif cmd == "ADD":
                if "TO WATCHLIST" in args:
                    params = self._parse_params(args.replace("TO WATCHLIST", ""))
                    if self.current_watchlist:
                        self.current_watchlist.add_to_watchlist(params["symbol"], params.get("notes", ""))
                    else:
                        print("No watchlist loaded. Use CREATE WATCHLIST first.")
                else:
                    print("Invalid ADD command. Use ADD TO WATCHLIST.")

            elif cmd == "RUN":
                if "STRATEGY" in args:
                    params = self._parse_params(args.replace("STRATEGY", ""))
                    if self.current_portfolio:
                        strategy_name = params["name"].lower()
                        if strategy_name == "bollinger bands":
                            self.current_portfolio.run_backtest(
                                self.current_portfolio.bollinger_bands_strategy,
                                symbol=params["symbol"],
                                start_date=params["start_date"],
                                end_date=params["end_date"]
                            )
                        else:
                            print(f"Strategy '{strategy_name}' not supported.")
                    else:
                        print("No portfolio loaded. Use CREATE PORTFOLIO or LOAD PORTFOLIO first.")
                else:
                    print("Invalid RUN command. Use RUN STRATEGY.")

            elif cmd == "GENERATE":
                if "ADVICE" in args:
                    params = self._parse_params(args.replace("ADVICE", ""))
                    generate_advice_sheet(params["symbol"])
                else:
                    print("Invalid GENERATE command. Use GENERATE ADVICE.")

            elif cmd == "SAVE":
                if self.current_portfolio:
                    save_portfolio(user_id, self.current_portfolio)
                elif self.current_watchlist:
                    save_watchlist(user_id, self.current_watchlist)
                else:
                    print("Nothing to save. Create or load a portfolio/watchlist first.")

            elif cmd == "LOAD":
                if "PORTFOLIO" in args:
                    params = self._parse_params(args.replace("PORTFOLIO", ""))
                    self.current_portfolio = load_portfolio(user_id, int(params["id"]))
                elif "WATCHLIST" in args:
                    params = self._parse_params(args.replace("WATCHLIST", ""))
                    self.current_watchlist = load_watchlist(user_id, int(params["id"]))
                else:
                    print("Invalid LOAD command. Use LOAD PORTFOLIO or LOAD WATCHLIST.")

            elif cmd == "EXIT":
                print("Exiting QuarkScript.")
                return True

            else:
                print(f"Unknown command: {cmd}")

        except Exception as e:
            print(f"Error executing command: {e}")

    def _parse_params(self, args):
        """Parse key-value pairs from command arguments."""
        params = {}
        for pair in args.split():
            if "=" in pair:
                key, value = pair.split("=")
                params[key.strip()] = value.strip().strip('"')
        return params


# --- Main Loop ---
def quark_script_main():
    interpreter = QuarkScriptInterpreter()
    print("Welcome to QuarkScript! Type your commands below.")
    while True:
        try:
            command = input("QuarkScript> ")
            if interpreter.execute(command):
                break
        except KeyboardInterrupt:
            print("\nExiting QuarkScript.")
            break


# Run the QuarkScript interpreter
quark_script_main()
