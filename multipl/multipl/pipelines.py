# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import datetime
import json

import dateutil.parser
from scrapy.exceptions import DropItem


class ValidatePipeline(object):
    def process_item(self, item, spider):
        try:
            dateutil.parser.parse(item['date']).date()
            return item
        except:
            raise DropItem("Not valid date")

class JsonWriterPipeline(object):

    def __init__(self):
        today = str(datetime.datetime.now().date()).replace("-", "_")
        fn = '../data/multipl_{0}.jl'.format(today)
        self.file = open(fn, 'wb')

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item