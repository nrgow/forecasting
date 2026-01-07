import logging

import fire
from dotenv import load_dotenv

from src.processing.run_pipeline import run_pipeline


class CLI:
    def run_pipeline(self) -> None:
        run_pipeline()


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    fire.Fire(CLI)
