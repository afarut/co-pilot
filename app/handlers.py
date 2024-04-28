from aiogram import F, Router
from aiogram.types import Message, CallbackQuery
from aiogram.filters import CommandStart
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
import app.keyboards as kb
import app.database.requests as rq
from model import CoPilot

copilot = CoPilot()
router = Router()
SECRET_KEY = "ПРИЧИНА ТРЯСКИ СЕКРЕТНЫЙ КЛЮЧ"

open_questions = {}
curator_ids = set()


class Key(StatesGroup):
    key = State()


class User_question(StatesGroup):
    question = State()


class Answering(StatesGroup):
    answer = State()


async def cmd_start(message: Message):
    await rq.set_user(message.from_user.id)
    await message.answer('Добро пожаловать во второй пилот для куратора / специалиста поддержки!', reply_markup=kb.main)


@router.message(CommandStart())
async def start_command(message: Message):
    await cmd_start(message)


@router.message(F.text == 'Я студент')
async def is_student(message: Message, state: FSMContext):
    if message.from_user.id in curator_ids:
        curator_ids.remove(message.from_user.id)
    await message.answer('Задайте свой вопрос:')
    await state.set_state(User_question.question)


@router.message(User_question.question)
async def handle_student_question(message: Message, state: FSMContext):
    await state.update_data(question=message.text)
    question = await state.get_data()
    result = copilot(question["question"], machine_state=False)
    if result["class"] >= 0:
        await message.answer(f"Результат найден. Класс: {result['class']}, Класс-метка: {result['class_label']}")
    elif result["class"] == -1:
        await message.answer("Нужен дополнительный вопрос.")
    else:
        open_questions[message.from_user.id] = question["question"]
        await message.answer("Ваш вопрос передан куратору. Ожидайте ответа.")
    await state.clear()


@router.message(F.text == 'Я куратор')
async def is_curator(message: Message, state: FSMContext):
    await message.answer('Введите свой ключ куратора')
    await state.set_state(Key.key)


@router.message(Key.key)
async def handle_curator_key(message: Message, state: FSMContext):
    await state.update_data(key=message.text)
    key = await state.get_data()
    if key["key"] == SECRET_KEY:
        curator_ids.add(message.from_user.id)
        await message.answer('Добро пожаловать, куратор! Чтобы ответить на вопрос напишите сообщение в формате: "Ответ на вопрос от пользователя [user_id]: [текст вашего ответа]"', reply_markup=kb.kurator_menu)
    else:
        await message.answer('Такого ключа не существует!')
    await state.clear()


@router.callback_query(F.data == 'kurator-q')
async def show_open_questions(callback: CallbackQuery):
    if open_questions:
        await callback.message.answer('Вопросы для куратора', reply_markup=kb.answer)
        for user_id, question in open_questions.items():
            await callback.message.answer(f"Пользователь {user_id}: {question}")
    else:
        await callback.message.answer('Открытых вопросов нет.')


@router.callback_query(F.data.startswith('answer-'))
async def answer_to_user(callback: CallbackQuery):
    user_id = int(callback.data.split('-')[1])
    question = open_questions.get(user_id)
    await Answering.answer.set()
    await callback.message.answer(f"Отвечаете на вопрос от пользователя {user_id}: {question}")


@router.message(lambda message: message.text.startswith('Ответ на вопрос от пользователя'))
async def handle_answer_to_user(message: Message, state: FSMContext):
    if message.from_user.id in curator_ids:
        parts = message.text.split(':')
        if len(parts) == 2:
            user_id = parts[0].split()[-1]
            text = parts[1].strip()
            question_user_id = int(user_id)
            if question_user_id in open_questions:
                await message.bot.send_message(question_user_id, f"Куратор ответил на ваш вопрос: {text}")
                del open_questions[question_user_id]
                await message.answer(f"Вы ответили на вопрос пользователя {question_user_id}.")
            else:
                await message.answer(f"Вопрос от пользователя {question_user_id} не найден.")
        else:
            await message.answer("Неверный формат ответа.")
    else:
        await message.answer("Для ответа на вопросы пользователей требуется правильный ключ куратора.")