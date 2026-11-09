"""Application entrypoint to launch the Uvicorn ASGI server."""

import uvicorn
from src.config import settings
from src.logging_config import logger

def main():
    """Launch the AskYourDoc ASGI application server."""
    logger.info("Starting AskYourDoc server on %s:%s", settings.HOST, settings.PORT)
    uvicorn.run(
        "src.api.app:create_app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        reload_dirs=["src", "templates", "static"],
        reload_excludes=[".venv", "data", "logs", "*.log", "*.db"],
        factory=True
    )


if __name__ == "__main__":
    main()
