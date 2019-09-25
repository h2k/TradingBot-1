class Stock(object):
    """Nasdaq Stock"""
    def __init__(self, values):
        self.symbol = values[0]
        self.name = values[1]
        try:
            self.price = float(values[2])
        except ValueError:
            self.price = 9999
        self.entry_year = values[4]
        self.sector = values[5]
        self.industry = values[6]
        self.risk = 0
        self.quantity_to_buy = 0

    def to_json(self):
        return {"symbol": self.symbol, "name": self.name, "price": self.price, "sector": self.sector, "industry": self.industry,
                "entry_year": self.entry_year}

    @staticmethod
    def from_nasdaq(values):
        return Stock(values)
