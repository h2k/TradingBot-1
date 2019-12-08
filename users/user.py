import requests


class User(object):
    def __init__(self, bearer):
        self.pooltrade_api_url = "https://api-pooltrade.ngrok.io"
        self.bearer = bearer
        self.headers = {}
        self.balance = 0
        self.invested_balance = 0
        self.risk = "low"
        self.inventory = []
        if bearer is not None:
            self.load_user(bearer)

    def create_user(self, first_name, last_name, email, password):
        """
        Create a user (useful for test purpose)
        """
        data = {
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password": password
        }
        r = requests.post(f'{self.pooltrade_api_url}/user', json=data)
        return r.ok

    def login(self, email, password):
        """
        Login and retrieve a bearer token
        """
        data = {
            "email": email,
            "password": password
        }
        r = requests.post(f'{self.pooltrade_api_url}/login', json=data)
        if r.ok:
            self.bearer = r.json()["access_token"]
            self.load_user(self.bearer)

    def load_user(self, bearer):
        """
        Init all the values concerning a user
        """
        self.headers = {
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json"
        }
        self.balance = self.get_balance()
        self.invested_balance = self.get_invested_balance()
        self.risk = self.get_risk()
        self.inventory = self.get_inventory()

    def get_balance(self):
        """
        Retrive the user total balance
        """
        r = requests.get(f'{self.pooltrade_api_url}/general_balance', headers=self.headers)
        if r.ok:
            return float(r.json()["amount"])

    def update_balance(self, new_balance):
        """
        Update the user balance
        """
        self.balance = new_balance
        data = {
            "amount": self.balance
        }
        r = requests.post(f'{self.pooltrade_api_url}/general_balance', json=data, headers=self.headers)
        return r.ok

    def get_invested_balance(self):
        """
        Retrive the user invested balance
        """
        r = requests.get(f'{self.pooltrade_api_url}/invested_balance', headers=self.headers)
        if r.ok:
            return float(r.json()["amount"])

    def update_invested_balance(self, new_balance):
        """
        Update the user invested balance
        """
        self.invested_balance = new_balance
        data = {
            "amount": self.invested_balance
        }
        r = requests.post(f'{self.pooltrade_api_url}/invested_balance', json=data, headers=self.headers)
        if r.ok:
            # Retrive balance because invested is taken out of balance
            self.balance = self.get_balance()
        return r.ok

    def get_risk(self):
        """
        Retrieve the user risk index (low, medium, high)
        """
        r = requests.get(f"{self.pooltrade_api_url}/risk_index", headers=self.headers)
        if r.ok:
            return r.json()["value"]

    def get_inventory(self):
        """
        Retrive user inventory
        """
        r = requests.get(f"{self.pooltrade_api_url}/owned_shares?size=100&page=0", headers={
          f"Authorization": f"Bearer {self.bearer}",
        })
        if r.ok:
            return r.json()["stocks"]
        return []

    def update_inventory(self, symbol, quantity, income, action):
        """
        Retrive user inventory

        :param symbol Stock symbol
        :param quantity Quantity bought
        :param income Close price at T
        :param action Buy or sell
        """
        data = {
            "symbol": symbol,
            "quantity": quantity,
            "income": income,
            "action": action
        }
        r = requests.post(f"{self.pooltrade_api_url}/owned_shares", json=data, headers=self.headers)
        if r.ok:
            self.inventory = self.get_inventory()
            self.invested_balance = self.get_invested_balance()
            self.balance = self.get_balance()