class Stock(object):
    """Nasdaq Stock"""
    def __init__(self, values):
        self.symbol = values[0]
        self.name = values[1]
        try:
            self.price = float(values[2])
        except ValueError:
            self.price = 9999
        if values[5] == "n/a":
            self.entry_year = "2015-01-01"
        else:
            self.entry_year = f'{values[5]}-01-01'
        self.sector = values[6]
        self.industry = values[7]
        self.risk = 0
        self.quantity_to_buy = 0
        self.is_crypto = False

    @staticmethod
    def from_nasdaq(values):
        return Stock(values)
