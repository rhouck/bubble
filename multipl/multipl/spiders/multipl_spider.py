import scrapy

from multipl.items import MultiplItem


class DmozSpider(scrapy.Spider):
    name = "multipl"
    allowed_domains = ["multpl.com"]
    start_urls = [
        #"http://www.multpl.com/s-p-500-book-value/table/by-quarter",
        "http://www.multpl.com/sitemap",
    ]

    def parse_contents(self, response):
        
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

    def parse(self, response):
        
        hrefs = []
        for i in (1,2):
            r = response.xpath('//div[@class="col{0}"]'.format(i)).xpath('.//li/a/@href').extract()
            for href in r:
                hrefs.append(href)
        href = set(hrefs)

        def build_urls(href):
            base = "http://www.multpl.com/"
            href = href.replace("/", "") 
            href = href + "/" if href else ""         
            a = base + href + 'table/?f=m'
            b = base + href + 'table/by-quarter/'
            return a, b
        
        hrefs = [build_urls(h) for h in hrefs]
        hrefs = [item for sublist in hrefs for item in sublist]
        
        for url in hrefs:
            yield scrapy.Request(url, callback=self.parse_contents)



