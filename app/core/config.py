"""app/core/config.py"""
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME:    str  = "WAF-ML"
    APP_VERSION: str  = "1.0.0"
    DEBUG:       bool = False

    MONGO_URI: str = "mongodb://mongodb:27017"
    MONGO_DB:  str = "waf_db"

    L2A_ONNX_PATH:      Path = Path("ml/exported_models/layer2a_best.onnx")
    L2A_THRESHOLD_PATH: Path = Path("ml/exported_models/layer2a_best_threshold.txt")
    L2B_ONNX_PATH:      Path = Path("ml/exported_models/layer2b_best.onnx")
    SCALER_PATH:        Path = Path("ml/exported_models/scaler_l2a.pkl")

    SCORE_LOG_THRESHOLD:   int = 30
    SCORE_BLOCK_THRESHOLD: int = 70
    RATE_LIMIT_PER_MIN:    int = 100
    PROTECTED_APP_URL:     str = "http://webapp:5000"
    HEALTH_CHECK_INTERVAL_SEC: int   = 60
    ERROR_RATE_THRESHOLD:      float = 0.10
    RETRAIN_MIN_SAMPLES:       int   = 200

    # L2B latency control
    L2B_TOKEN_MAX_LEN:     int = 256  # must match feature_extractor.py CharTokenizer max_len

    # Rate-limiter / IP-ban settings
    IP_BAN_THRESHOLD:      int = 5    # violations before IP is auto-banned
    IP_BAN_DURATION_SEC:   int = 300  # ban duration in seconds (5 minutes)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()