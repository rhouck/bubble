import scrapy

from multipl.items import MultiplItem


class DmozSpider(scrapy.Spider):
    name = "multipl"
    allowed_domains = ["multpl.com"]
    start_urls = [
        "http://www.multpl.com/s-p-500-book-value/table/by-quarter",
    ]

    def parse(self, response):
        
        clean = lambda x: x.split("\n")[0].strip()
        title = response.xpath('//table[@id="datatable"]/tr/th/span[@class="title"]/text()').extract()[0]
        for i in ('odd', 'even'):
            rows = response.xpath('//table[@id="datatable"]/tr[@class="{0}"]'.format(i))
            for r in rows:
                a, b = r.xpath('.//td/text()').extract()
                item = MultiplItem()
                item['title'] = clean(title)
                item['date'] = clean(a)
                item['value'] = clean(b)
                yield item



