import re
import subprocess
from lxml import etree
import csv
def fetch_pubmed_data(pubmed_id):
    """使用 EDirect 获取指定 PubMed ID 的数据，并解析 XML"""
    command = f"efetch -db pubmed -id {pubmed_id} -format xml"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout






def parse_pubmed_xml(xml_data):

    xml_data = xml_data.replace('&rsquo;', '&#8217;')
    xml_data = re.sub('<CoiStatement>.*?</CoiStatement>', '', xml_data, flags=re.DOTALL)
    xml_bytes = xml_data.encode('utf-8')
    root = etree.fromstring(xml_bytes)



    pmid = root.find('.//PMID').text if root.find('.//PMID') is not None else 'unknown'


    pub_date = root.find('.//PubDate/Year').text if root.find('.//PubDate/Year') is not None else 'unknown'


    beginning_date = root.find('.//BeginningDate/Year').text if root.find(
        './/BeginningDate/Year') is not None else 'unknown'


    ending_date = root.find('.//EndingDate/Year').text if root.find('.//EndingDate/Year') is not None else 'unknown'

    article_title = root.find(".//ArticleTitle").text if root.find('.//ArticleTitle') is not None else 'unknown'


    authors = []
    author_list = root.find('.//AuthorList[@Type="authors"]')
    if author_list is None:
        author_list = root.find('.//AuthorList')

    if author_list is not None:
        authors = [
            f"{author.find('LastName').text} {author.find('ForeName').text}"
            for author in author_list.findall('.//Author')
            if author.find('LastName') is not None and author.find('ForeName') is not None
        ]

    if not authors:
        authors = 'unknown'


    authors_str = ', '.join(authors)


    abstract_texts = root.findall('.//Abstract/AbstractText')
    abstract = ' '.join([abstract_text.text for abstract_text in abstract_texts if abstract_text.text])


    keywords_texts = root.findall('.//KeywordList/Keyword')
    keywords = ', '.join([keywords_text.text for keywords_text in keywords_texts if keywords_text.text])
    if not keywords:
        keywords = 'unknown'

    return {
        'PMID': pmid,
        'PubDate': pub_date,
        'BeginningDate': beginning_date,
        'EndingDate': ending_date,
        'Authors': authors_str,
        'Abstract': abstract,
        'Keywords': keywords,
        'ArticleTitle': article_title,
    }


input_file = './output_file.txt'
output_file = 'latest/pubmed_dataset_只有2.csv'
start_line = 1
end_line = 62477



def write_row_to_csv(csv_writer, article_data):
    authors_str = article_data['Authors']
    row = [article_data['PMID'], article_data['PubDate'], article_data['BeginningDate'],
           article_data['EndingDate'], authors_str, article_data['Abstract'], article_data['Keywords'],
           article_data['ArticleTitle']]
    csv_writer.writerow(row)


from tqdm import tqdm

errorId = []
# Open the CSV file in append mode
with open(input_file, 'r') as ids_file, open(output_file, 'w', newline='', encoding='utf-8') as data_file:
    csv_writer = csv.writer(data_file)

    # Check if the file is empty to decide whether to write headers
    data_file.seek(0, 2)  # Go to the end of the file
    if data_file.tell() == 0:  # If file is empty, write headers
        headers = ['PMID', 'PubDate', 'BeginningDate', 'EndingDate', 'Authors', 'Abstract', 'Keywords', 'ArticleTitle']
        csv_writer.writerow(headers)

    for _ in range(start_line - 1):
        next(ids_file)

    for _ in tqdm(range(end_line - start_line + 1), desc="Processing PubMed IDs"):
        pubmed_id = next(ids_file).strip()
        xml_data = str(fetch_pubmed_data(pubmed_id))
        article_data = parse_pubmed_xml(xml_data)
        write_row_to_csv(csv_writer, article_data)


