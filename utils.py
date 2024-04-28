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


def get_summars(file_path):
  df = pd.read_csv(file_path)
  data = [0] * len(df)
  answer_class = df["answer_class"]
  summ = df["summ"]
  for i in range(len(df)):
      data[answer_class[i]] = summ[i]
  return data


def get_answers(file_path):
  df = pd.read_csv(file_path)
  data = [0] * len(df)
  answer_class = df["answer_class"]
  summ = df["Answer"]
  for i in range(len(df)):
      data[answer_class[i]] = summ[i]
  return data
