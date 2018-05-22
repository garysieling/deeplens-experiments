report = """                                precision    recall  f1-score   support

            Acadian_Flycatcher       0.00      0.00      0.00        23
              Acorn_Woodpecker       1.00      0.12      0.21        25
              Zone_tailed_Hawk       0.00      0.00      0.00        17

                   avg / total       0.02      0.01      0.01     11258"""
import re

report = [
  re.compile(" [ ]+").split(x.strip())
  for x in report.split("\n")[2:] 
  if len(x) > 0
]

import csv

with open('test.csv', 'w') as csvfile:
  csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  csvwriter.writerow(['build number', 'label', 'precision', 'recall', 'f1-score', 'support'])

  [
    csvwriter.writerow(['1'] + x) for x in report
  ]
