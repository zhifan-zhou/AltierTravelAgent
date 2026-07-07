"""FastAPI backend for the AltierTravelAgent web prototype."""

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    if name in __all__:
        from travel_agent.server.app import app, create_app

        return {"app": app, "create_app": create_app}[name]
    raise AttributeError(name)
