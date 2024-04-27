import torch
from torch.nn.functional import softmax
from utils import get_categories
from classifier import RubertTinyClassifier
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")


class CoPilot:
	def __init__(self, alpha=0.00003, e=0.000001):
		self.e = e
		self.alpha = alpha

		if self.e >= self.alpha:
			raise Exception("Ошибка установки констант, self.alpha должен быть больше e")

		self.categories = get_categories("./datasets/orig_train_data.csv")
		self.categoriy_label = [
		 "Документы", 
		 "Организация уроков", 
		 "Портал", 
		 "Трудоустройство", 
		 "Программа обучения", 
		 "Расписание", 
		 "Требования ПО", 
		 "Перевод/ запись в группу", 
		 "Оценки", 
		 "Практические работы"]

		
		self.answer_classifier = RubertTinyClassifier(30)
		self.answer_classifier.load_state_dict(torch.load("./weights/RuBertAnswer.pt"))

		self.category_classifier = RubertTinyClassifier(10)
		self.category_classifier.load_state_dict(torch.load("./weights/RuBertCategory.pt"))
		self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

	def __call__(self, question: str, machine_state: bool=True) -> int:
		# machine state доп вопрос
		result = -1

		# class == -1 Доп вопрос
		# class == -2 К куратору
		# Иначе предсказанный класс

		data = self.tokenizer(question, padding=True, truncation=True, return_tensors='pt')
		
		answer_logits = self.answer_classifier(**data).squeeze()
		category_logits = self.category_classifier(**data).squeeze()

		answer_distr = softmax(answer_logits)
		category_distr = softmax(category_logits)

		print(answer_distr.round(decimals=2))
		print(category_distr.round(decimals=2))

		
		if torch.max(answer_distr).item() < self.alpha or torch.max(category_distr).item() < self.alpha:
			if machine_state:
				return {"status": "random", "text": "Нужен доп вопрос", "class": -1}
			else:
				return {"status": "error", "text": "К куратору", "class": -2}
		else:
			category_id = category_distr.argmax()
			min_answer = torch.max(answer_distr).item() - self.e
			answers = []
			for i in range(len(answer_distr)):
				if answer_distr[i].item() > min_answer:
					answers.append((i, answer_distr[i].item()))
			answers.sort(key=lambda x: x[1], reverse=True)
			print("INFO:", answers)
			print("INFO:", self.categoriy_label[category_id])
			if len(answers) == 1:
				if answers[0][0] in self.categories[self.categoriy_label[category_id]]:
					return {"status": "ok", 
							"text": "Результат найден", 
							"class": answers[0][0], 
							"class_label": answers[0][1]}
				else:
					if machine_state:
						return {"status": "random", "text": "Нужен доп вопрос, Категория не соответствует клаасу ответа", "class": -1}
					else:
						return {"status": "error",
								"text": "Предсказанная категория не соответствует клаасу ответа. К куратору",
								"class": -2}

			for id, answer in answers:
				if id in self.categories[self.categoriy_label[category_id]]:
					break
			else:
				if machine_state:
					return {"status": "random", "text": "Нужен доп вопрос, Категории не соответствует классам ответов", "class": -1}
				else:
					return {"status": "error", "text": "Категория не соответствует классам ответов. К куратору", "class": -2}
			return {"status": "random", "text": "Среди предсказанных классов ответов есть нужный нам, но он не один", "class": -1}


if __name__ == "__main__":
	model = CoPilot()
	print("---------------")
	s = input()
	data = model(s)
	while data["class"] < 0:
		print("text:", data["text"], "class:", data["class"])
		s = s + " " + input()
		data = model(s, machine_state=False)

	print("text:", data["text"], "class:", data["class"], data["class_label"])
	print(model)