from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_async_engine(url=os.getenv("DATABASE_URL"))
AsyncSessionLocal = async_sessionmaker(bind=engine, autoflush=False)


async def get_session():
    async with AsyncSessionLocal() as session:
        yield session
