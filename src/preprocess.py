import re
import pandas as pd
import json


class DataClean(object):

    def __init__(self):
        """Initial some constants"""
        self.configuration_path = '/home/ubuntu/Project/EventExtraction/config/dataClean.json'
        self.config = self._load_config()
        self.special_chars = self._special_chars()
        self.startwith_blanks = self._startwith_blanks()
        self.remove_prefixes = self._remove_prefixes()

    def _load_config(self):
        """Load configurations from config file."""
        cfg_dict = {}
        with open(self.configuration_path, 'r') as cfg:
            cfg_dict = json.loads(cfg.read())
        return cfg_dict

    def _special_chars(self):
        """Get all special characters."""
        return [line.strip() for line in open(self.config['special_chars'], 'r')]

    def _startwith_blanks(self):
        return tuple([line.strip() for line in open(self.config['startwith_blanks'], 'r')])

    def _remove_prefixes(self):
        return [line.strip() for line in open(self.config['remove_prefix'])]

    def preprocess_line(self, line, sep='\t') -> list:
        """Clean data line by line"""
        # remove zero-width spaces
        line = line.replace('\u200b', '')
        try:
            title, time_str = tuple(filter(None, line.split(sep)))
        except:
            return []
        # remove too short lines
        if len(title) < self.config['MIN_TITLE_LENGTH']:
            return []
        if any(char in title for char in self.special_chars):
            return []
        if title.startswith(self.startwith_blanks):
            return []
        for prefix in self.remove_prefixes:
            if title.startswith(prefix):
                title = title[len(prefix):]
                break
        if " " in title:
            title = re.sub(r"[ ]*([a-zA-Z0-9]+)[ ]*", r"\1", title)
        lines = list(filter(None, re.split(r'，|：| |；', title)))
        # FIXME: function
        if len(lines) != 0 and lines[0] in ["外媒称", "外媒", "路透社", "彭博", "彭博社", "艾媒《外卖年度报告》", "图文", "业内人说", "万众期待", "快讯"]:
            lines = lines[1:]
        res_lines = []
        if len(lines) > 2:
            for index, l in enumerate(lines[:-1]):
                smaple_dict = {}
                smaple_dict["text"] = re.sub(',,', ',', ','.join([str(l), str(lines[index + 1]), time_str]))
                smaple_dict['labels'] = []
                res_lines.append(json.dumps(smaple_dict) + '\n')
        else:
            smaple_dict = {}
            lines.append(time_str)
            smaple_dict["text"] = re.sub(',,', ',', ','.join([str(l) for l in lines]))
            smaple_dict['labels'] = []
            res_lines.append(json.dumps(smaple_dict) + '\n')
        return res_lines

    def run(self):
        input_file = open(self.config['input_file'], 'r', encoding='utf-8')
        output_file = open(self.config['output_file'], 'w', encoding='utf-8')
        for line in input_file:
            line = self.preprocess_line(line)
            if len(line) > 0:
                output_file.writelines(line)
        input_file.close()
        output_file.close()
