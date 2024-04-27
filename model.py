import torch
from torch.nn.functional import softmax
from utils import get_categories, get_summars
from classifier import RubertTinyClassifier
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")


class CoPilot:
	def __init__(self, alpha=0.01, e=0.2):
		self.e = e
		self.alpha = alpha

		

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

		self.summars = get_summars("./datasets/summans.csv")

		if torch.cuda.is_available():
			self.device = "cuda"
		else:
			self.device = "cpu"

		
		self.answer_classifier = RubertTinyClassifier(30).to(self.device).eval()
		self.answer_classifier.load_state_dict(torch.load("./weights/RuBertAnswer.pt"))

		self.category_classifier = RubertTinyClassifier(10).to(self.device).eval()
		self.category_classifier.load_state_dict(torch.load("./weights/RuBertCategory.pt"))
		self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")

		self.mistral = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").to(self.device)
		self.mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")



	def __call__(self, question: str, machine_state: bool=True) -> int:
		# machine state доп вопрос
		result = -1

		# class == -1 Доп вопрос
		# class == -2 К куратору
		# Иначе предсказанный класс

		data = self.tokenizer(question, padding=True, truncation=True, return_tensors='pt').to(self.device)
		
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
			answer_summars = []
			
			for id, answer in answers:
				answer_summars.append(self.summars[id])
			result_str = "\"" + "\", \"".join(answer_summars[:-1]) + "\""
			result_str = f"{result_str}, и {answer_summars[-1]}\""
			promt = f'студент задает вопрос в бот обратной связи.\n вопрос студента: "{question}",\nварианты ответов: {result_str}\nв Напиши уточняющий вопрос к вопросу студента, Чтобы понять что конкретно его интересует. \n Ты задаешь вопрос от лица чат бота студенту для разьяснения того, какой ответ ему предоставить'
			messages = [{"role": "user", "content": promt},]
			encodeds = self.mistral_tokenizer.apply_chat_template(messages, return_tensors="pt")
			model_inputs = encodeds.to(self.device)
			generated_ids = self.mistral.generate(model_inputs, max_new_tokens=30, do_sample=True, pad_token_id=self.mistral_tokenizer.pad_token_id, eos_token_id=self.mistral_tokenizer.eos_token_id)
			decoded = self.mistral_tokenizer.batch_decode(generated_ids)
			text = decoded[0]
			return {"status": "random", "text": text, "class": -1}


if __name__ == "__main__":
  try:
    while 1:
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
  except:
    pass
