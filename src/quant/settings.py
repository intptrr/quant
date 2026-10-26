from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Field(default=Path("data"), validation_alias="QUANT_DATA_DIR")
    sharadar_api_key: SecretStr | None = Field(default=None, validation_alias="SHARADAR_API_KEY")
    alpaca_api_key: SecretStr | None = Field(default=None, validation_alias="ALPACA_API_KEY")
    alpaca_secret_key: SecretStr | None = Field(default=None, validation_alias="ALPACA_SECRET_KEY")
    ibkr_host: str = Field(default="127.0.0.1", validation_alias="IBKR_HOST")
    ibkr_port: int = Field(default=4002, validation_alias="IBKR_PORT")
    ibkr_client_id: int = Field(default=17, validation_alias="IBKR_CLIENT_ID")
    ibkr_account: str | None = Field(default=None, validation_alias="IBKR_ACCOUNT")
