import json

INPUT_FILE = '/home/ubuntu/Project/EventExtraction/data/Preprocess/clean.txt'
OUTPUT_FILE = '/home/ubuntu/Project/EventExtraction/data/Preprocess/news.csv'

if __name__ == '__main__':
    lines = open(INPUT_FILE, 'r').readlines()
    with open(OUTPUT_FILE, 'w') as w:
        w.write('title\ttime\n')
        for line in lines:
            line_js = json.loads(line)
            text = line_js['text']
            text, timestamp = text.split(',')[:-1], text.split(',')[-1]
            text = ','.join(text)
            w.write("{}\t{}".format(text, timestamp))