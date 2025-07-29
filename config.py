from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""

    ALGO_USED: str = Field(
        "geo_bleu",
        description="Algorithm used for calculations",
        choices=["geo_bleu", "dtw"],
    )

    TEST_USERS: int = Field(3000, description="Test users per dataset")

    TIMESTAMPS_PER_DAY: int = Field(48, description="Number of timestamps per day")
    ALL_DAYS: int = Field(75, description="Total number of days in the dataset")
    TRAIN_DAYS: int = Field(60, description="Number of days to train the model")
    TEST_DAYS: int = Field(15, description="Number of days to test the model")

    MIN_X: int = Field(1, description="Minimum value for X")
    MAX_X: int = Field(200, description="Maximum value for X")
    MIN_Y: int = Field(1, description="Minimum value for Y")
    MAX_Y: int = Field(200, description="Maximum value for Y")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
