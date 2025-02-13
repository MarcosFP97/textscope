import json
from textscope.relevance_analyzer import RelevanceAnalyzer
from textscope.subtheme_analyzer import SubthemeAnalyzer

rel = RelevanceAnalyzer()
subtheme = SubthemeAnalyzer()

with open('news_articles_ciber.jsonl', 'r') as file:
    data = file.readlines()
    print(len(data))
    for line in data:
        dd = json.loads(line)
        news = dd['text']
        score = rel.analyze(news, "cibersec")
        dd["score_rel"] = score
        puncts = subtheme.analyze_bin(news, "cibersec", thr=84.5)
        dd["subthemes"] = puncts
        with open('news_articles_ciber_scores.jsonl', 'a+') as f:
            f.write(json.dumps(dd)+'\n')

with open('x_posts_ciber.jsonl', 'r') as file:
    data = file.readlines()
    for line in data:
        tweet = json.loads(line)
        text = tweet["full_text"]
        score = rel.analyze(text, "cibersec")
        tweet["score_rel"] = score
        puncts = subtheme.analyze(text, "cibersec")
        tweet["subthemes"] = puncts
        with open('x_posts_ciber_scores.jsonl', 'a+') as f:
            f.write(json.dumps(tweet)+'\n')