import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

type FloatArray = npt.NDArray[np.float64]


class BrokerName(StrEnum):
    ALPACA = "alpaca"
    IBKR = "ibkr"


class BrokerTerms(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    borrow_fee: float = Field(default=0.0, ge=0)
    margin_spread: float = Field(default=0.0, ge=0)
    cash_spread: float | None = Field(default=None, ge=0)
    cash_threshold: float = Field(default=0.0, ge=0)
    full_rate_nav: float = Field(default=0.0, ge=0)
    commission_per_share: float = Field(default=0.0, ge=0)
    commission_minimum: float = Field(default=0.0, ge=0)
    commission_maximum: float = Field(default=0.01, ge=0)
    fees_per_share: float = Field(default=0.0, ge=0)
    sale_fee_rate: float = Field(default=0.0, ge=0)

    def commissions(self, quantities: FloatArray, prices: FloatArray) -> FloatArray:
        shares = np.abs(quantities)
        value = shares * prices
        broker = np.minimum(
            np.maximum(self.commission_minimum, self.commission_per_share * shares),
            self.commission_maximum * value,
        )
        sales = np.where(quantities < 0.0, self.sale_fee_rate * value, 0.0)
        return broker + self.fees_per_share * shares + sales

    def debit_rate(self, benchmark: FloatArray) -> FloatArray:
        return benchmark + self.margin_spread

    def credit_rate(self, benchmark: FloatArray, nav: float) -> FloatArray:
        if self.cash_spread is None:
            return np.zeros_like(benchmark)
        scale = min(nav / self.full_rate_nav, 1.0) if self.full_rate_nav else 1.0
        return np.maximum(benchmark - self.cash_spread, 0.0) * scale


class TradingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    broker: BrokerName = BrokerName.IBKR
    capital: float = Field(default=100_000.0, gt=0)
    min_order_value: float = Field(default=0.0, ge=0)
    max_order_fraction: float = Field(default=0.75, gt=0)
    max_price_deviation: float = Field(default=0.1, gt=0)
    max_drawdown: float = Field(default=0.25, gt=0, lt=1)
    slippage_bps: float = Field(default=3.0, ge=0)
    brokers: dict[BrokerName, BrokerTerms] = Field(default_factory=dict)

    @model_validator(mode="after")
    def has_terms_for_the_broker(self) -> Self:
        if self.broker not in self.brokers:
            raise ValueError(f"no [brokers.{self.broker}] terms in the trading config")
        return self

    @property
    def terms(self) -> BrokerTerms:
        return self.brokers[self.broker]

    def using(self, broker: BrokerName) -> Self:
        return self.model_validate({**self.model_dump(), "broker": broker})


def load_trading_config(path: Path) -> TradingConfig:
    return TradingConfig.model_validate(tomllib.loads(path.read_text()))
