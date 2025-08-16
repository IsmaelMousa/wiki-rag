from uvicorn                 import run
from fastapi                 import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config                  import get_config
from routers                 import router

cfg     = get_config().app
host    = cfg.host
port    = cfg.port
path    = cfg.path
version = cfg.version

app     = FastAPI(version=version)

app.add_middleware(middleware_class =CORSMiddleware,
                   allow_origins    =["*"],
                   allow_credentials=True,
                   allow_methods    =["*"],
                   allow_headers    =["*"],)

app.include_router(router=router)

if __name__ == "__main__": run(app, host=host, port=port)