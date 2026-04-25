"""app/main.py — FastAPI application entry point"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config    import settings
from app.core.logging   import setup_logging, logger
from app.core.exceptions import (ModelNotLoadedError, DatabaseError,
                              model_not_loaded_handler, database_error_handler)
from app.db.mongodb     import connect_db, close_db
from app.middleware.waf_middleware import WAFMiddleware
from app.middleware.rate_limiter   import rate_limiter  # sliding-window limiter (replaces slowapi)

import app.services.layer1_filter as l1
import app.services.layer2a_anomaly as l2a
import app.services.layer2b_deep as l2b
from app.services.health_monitor import start_monitor, stop_monitor

from app.api.routes.traffic   import router as traffic_router
from app.api.routes.logs      import router as logs_router
from app.api.routes.feedback  import router as feedback_router
from app.api.routes.health    import router as health_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.models    import router as models_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(debug=settings.DEBUG)
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)

    try:
        await connect_db()
    except Exception as e:
        raise DatabaseError(str(e))

    try:
        l2a.load()
        l2b.load()
        logger.info("All ML models loaded successfully")
    except FileNotFoundError as e:
        logger.error("Model file missing: %s", e)
        raise ModelNotLoadedError(str(e))

    await start_monitor()
    logger.info("WAF ready ◈")

    yield

    await stop_monitor()
    await close_db()
    logger.info("WAF shutdown complete")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(WAFMiddleware)
app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
app.add_exception_handler(DatabaseError,       database_error_handler)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(traffic_router)   # already has prefix="/api/traffic"
app.include_router(health_router)    # already has prefix="/api/health"
app.include_router(logs_router)      # already has prefix="/api/logs"
app.include_router(feedback_router)  # already has prefix="/api/feedback"
app.include_router(models_router)    # already has prefix="/api/models"
app.include_router(dashboard_router) # serves /dashboard, /dashboard/logs, /dashboard/feedback

@app.get("/")
async def root():
    return {"service": settings.APP_NAME, "version": settings.APP_VERSION,
            "dashboard": "/dashboard", "docs": "/api/docs"}