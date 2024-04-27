import pandas as pd


def get_categories(file_path):
	df = pd.read_csv(file_path)
	cat = df["Category"]
	answer = df["answer_class"]
	result = {}
	for i in range(len(df)):
	    if cat[i] not in result:
	        result[str(cat[i])] = set()
	    result[str(cat[i])].add(answer[i])
	return result