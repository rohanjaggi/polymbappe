"""Application configuration."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment."""

    model_config = SettingsConfigDict(env_prefix="POLYMBAPPE_", env_file=".env", extra="ignore")

    random_seed: int = Field(default=20260611)
    data_dir: Path = Field(default=Path("data"))
    autotune_llm_model: str = Field(default="qwen3.5:9b")

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def outputs_data_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def configs_dir(self) -> Path:
        return Path("configs")
