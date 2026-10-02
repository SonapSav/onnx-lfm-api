import uvicorn

from .config import settings


def main() -> None:
    # Single worker: the model is loaded once per process and generation is
    # serialized, so multiple workers would just multiply memory use.
    uvicorn.run(
        "onnx_lfm_api.api:app",
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
