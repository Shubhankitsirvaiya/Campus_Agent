"""Create or rebuild the local academic operations database."""

import argparse
import asyncio

from app.config import PROJECT_ROOT, get_settings
from app.services.database import AcademicRepository


async def main(force: bool) -> None:
    settings = get_settings()
    repository = AcademicRepository(
        database_path=settings.absolute_path(settings.academic_db_path),
        seed_root=PROJECT_ROOT / "data" / "seed",
    )
    await repository.initialize(force_seed=force)
    print(f"Academic database ready: {repository.database_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the CampusPath local academic database.")
    parser.add_argument("--force", action="store_true", help="Rebuild only the configured academic database.")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.force))
