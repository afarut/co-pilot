from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from app.database.requests import get_categories, get_category_item

# Главная клавиатура
main = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text='Я куратор'), KeyboardButton(text='Я студент')],
    ],
    resize_keyboard=True,
    input_field_placeholder='Выберите пункт меню...'
)

kurator_menu = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Показать список вопросов', callback_data='kurator-q')]])

answer = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Список открытых вопросов:', callback_data='answer')]])