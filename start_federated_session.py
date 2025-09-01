from fedn import APIClient
from config import settings
import argparse


def main():
    parser = argparse.ArgumentParser(description="Start a FEDn session")
    parser.add_argument("--name", default="Test", help="Optional session name")
    args = parser.parse_args()

    if settings["IS_LOCAL"]:
        print("Running in local mode")
        client = APIClient(host=settings["DISCOVER_HOST"], port=8092)
    else:
        print("Running in remote mode")
        client = APIClient(
            host=settings["DISCOVER_HOST"],
            token=settings["CLIENT_TOKEN"],
            secure=settings["SECURE"],
            verify=settings["VERIFY"],
        )
    client.set_active_model("seed/seed.npz")
    trail = client.get_nmodel_trail()

    id = trail[-1]["id"]
    client.start_session(name=args.name, rounds=settings["ROUNDS"], model_id=id, round_timeout=settings["ROUND_TIMEOUT"])


if __name__ == "__main__":
    main()
