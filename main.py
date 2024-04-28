import asyncio
from aiogram import Bot, Dispatcher
from app.handlers import router
from app.database.models import async_main


async def main():
    await async_main()
    bot = Bot(token='6715118559:AAGK8Upr1nZzi_9IkOP6U3Sxx6PE64JhEl8')
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Бот выключен')
