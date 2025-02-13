import json
from textscope.relevance_analyzer import RelevanceAnalyzer
from textscope.subtheme_analyzer import SubthemeAnalyzer

rel = RelevanceAnalyzer()
subtheme = SubthemeAnalyzer()

with open('news_articles.jsonl', 'r') as file:
    data = file.readlines()
    print(len(data))
    for line in data:
        dd = json.loads(line)
        news = dd['text']
        score = rel.analyze(news, "digital_rights")
        dd["score_rel"] = score
        puncts = subtheme.analyze_bin(news, "digital_rights")
        dd["subthemes"] = puncts
        print(f'{news} - SCORE: {puncts}')
        with open('news_articles_scores.jsonl', 'a+') as f:
            f.write(json.dumps(dd)+'\n')

with open('x_posts.jsonl', 'r') as file:
    data = file.readlines()
    for line in data:
        tweet = json.loads(line)
        text = tweet["full_text"]
        score = rel.analyze(text, "digital_rights")
        tweet["score_rel"] = score
        puncts = subtheme.analyze_bin(text, "digital_rights")
        tweet["subthemes"] = puncts
        print(f'{text} - SCORE: {puncts}')
        with open('x_posts_scores.jsonl', 'a+') as f:
            f.write(json.dumps(tweet)+'\n')