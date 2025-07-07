import numpy as np

class EWMA:
    def __init__(self, halflife: float):
        self.halflife = halflife
        self.alpha = 1 - np.exp(np.log(0.5) / halflife)
        self.value_ = None
        self.n = 0

    def add(self, x: float):
        if self.value_ is None:
            self.value_ = x
        else:
            self.value_ = self.alpha * x + (1 - self.alpha) * self.value_
        self.n += 1

    def value(self):
        return 0.0 if self.n < 2 else self.value_

class ASQuoteEngine:
    def __init__(self, gamma: float, kappa_default: float, vol_halflife_sec: float, tick: float):
        self.gamma  = gamma
        self.kappa0 = kappa_default
        self.tick   = tick
        self.vol_est = EWMA(halflife=vol_halflife_sec)  # keeps σ²

    def update_vol(self, fair_p_new: float):
        logit = np.log(fair_p_new / (1 - fair_p_new))
        self.vol_est.add(logit)

    def kappa_from_book(self, order_book) -> float:
        # order_book.top_levels() should return list of (price, size)
        depth = sum(size / abs(price - order_book.mid)
                    for price, size in order_book.top_levels() if price != order_book.mid)
        return depth or self.kappa0

    def quotes(self, fair_p: float, q_pos: int, time_left: float, order_book) -> tuple:
        sigma2 = self.vol_est.value() or 0.0
        kappa  = self.kappa_from_book(order_book)

        # 1. reservation price
        r = fair_p - q_pos * self.gamma * sigma2 * time_left

        # 2. optimal spread
        delta = self.gamma * sigma2 * time_left \
                + (2 / self.gamma) * np.log1p(self.gamma / kappa)

        bid = max(0.0, min(1.0, r - 0.5 * delta))
        ask = max(0.0, min(1.0, r + 0.5 * delta))

        # 3. snap to allowed tick
        bid = np.floor(bid / self.tick) * self.tick
        ask = np.ceil (ask / self.tick) * self.tick
        return bid, ask 